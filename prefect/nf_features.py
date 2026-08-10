"""
Normal-form features, measured on the table itself (Phase 2b of TASK_3_PLAN.md).

This module is the ``X`` side of the architecture rule from Phase 2::

    Generator --> Iceberg tables (RAW DATA)  -->  nf_features.py  -->  X
         |                                          (sees ONLY the table)
         +------> Manifest (FDs, label, params) -------------------->  y

It takes a catalog/schema/table and a connection, and returns one feature row per column.
It never reads the manifest, and it is imported by **both** the training-set builder and
the Prefect pipeline, which is what removes the train/serve skew from finding 1.6.

The three families, and which leak each one replaces
----------------------------------------------------
| family                     | replaces                                                |
| :------------------------- | :------------------------------------------------------ |
| atomicity (value-based)    | `is_this_col_violating_1nf`, copied from the label      |
| atomicity (name-based)     | nothing - repeating groups had no feature at all        |
| key + dependency structure | `is_composite_key_part`, `table_has_partial_dependency` |

The last family is the one that matters most: 2NF and 3NF differ only in *which* column a
dependency hangs off, and until now no feature measured that at all - which is why the
generator artifact in finding 1.3 was free to fill the gap.

Cost
----
Two batched queries per table, not a loop over column pairs. Everything below - the UCC
search up to level 2 and every single-attribute dependency - comes out of one pairwise
distinct-count matrix, ``C(n,2)`` aggregates. Level 3 only runs when levels 1 and 2 found
no key at all, and is pruned twice over (decision **E1**).

Three budgets bound that, and none of them is silent - each is printed when it fires and
reported to the model as a feature, because "found no key" and "stopped looking" have to
stay distinguishable:

| budget                    | bounds  | feature                                          |
| :------------------------ | :------ | :----------------------------------------------- |
| ``SAMPLE_ROW_THRESHOLD``  | rows    | ``table_sampled``, ``table_sample_ratio``        |
| ``MAX_PROFILED_COLUMNS``  | columns | ``table_pair_coverage``                          |
| ``MAX_LEVEL_3_CANDIDATES``| triples | ``table_ucc_search_coverage``                    |

All three make the measurement *approximate*, which is the point rather than a concession:
exact dependencies on complete data determine the normal form outright, and there would be
nothing left for a model to infer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sqlalchemy.engine import Connection


# --------------------------------------------------------------------------------------
# Atomicity: thresholds and the free-text veto
# --------------------------------------------------------------------------------------

# Kept as a module constant, never inlined into an f-string: inside an f-string `{1,40}`
# is evaluated as the tuple `(1, 40)` and silently corrupts the regex. Backslash-free on
# purpose - RE2J in Trino has no lookahead either way.
LIST_VALUE_PATTERN = "^[^,;|.!?]{1,40}([,;|] ?[^,;|.!?]{1,40})+$"

# Share of non-null values that must look like a list before the column counts as
# non-atomic.
LIST_VALUE_MIN_SHARE = 0.8

# Mean length of the pieces *between* separators, not of the whole value.
#
# This replaces the old "mean value length <= 120" guard, which was measured to produce a
# false negative: a supplier row listing 80 part ids is 355 characters long and was
# therefore treated as prose, even though every piece of it is 4 characters. The guard is
# meant to separate prose from lists, and what separates them is the piece length - a
# genuine list stays short per element no matter how many elements it has.
LIST_TOKEN_MAX_MEAN_LENGTH = 20

# Fallback if the config file is missing. Free-text names only: audit columns
# (created_at, version, is_deleted) never trip the detector anyway.
DEFAULT_FREETEXT_NAME_TOKENS: tuple[str, ...] = (
    "bemerkung",
    "kommentar",
    "comment",
    "beschreibung",
    "description",
    "freitext",
    "freetext",
    "memo",
    "notiz",
    "note",
    "zusatzinfo",
    "feedback",
    "review",
)

_FREETEXT_CONFIG_PATH = Path(__file__).with_name("nf_freetext_columns.json")


def _load_freetext_tokens() -> tuple[str, ...]:
    """
    Read the free-text name tokens from nf_freetext_columns.json, with a built-in
    fallback. Config rather than code, because which columns hold prose is a property of
    the databases being profiled.
    """
    try:
        with _FREETEXT_CONFIG_PATH.open(encoding="utf-8") as handle:
            tokens = json.load(handle)["freetext_name_tokens"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Could not read {_FREETEXT_CONFIG_PATH.name} ({error}); using built-in defaults")
        return DEFAULT_FREETEXT_NAME_TOKENS
    return tuple(str(token).strip().lower() for token in tokens if str(token).strip())


FREETEXT_NAME_TOKENS = _load_freetext_tokens()


def is_freetext_name(column_name: str) -> bool:
    """True if the column name marks free-form prose (comment, note, description, ...)."""
    lowered = str(column_name).lower()
    return any(token in lowered for token in FREETEXT_NAME_TOKENS)


# ``phone_1``, ``phone2``, ``order_03`` - a repeating group is visible only in the names.
_REPEATING_SUFFIX = re.compile(r"^(?P<stem>.*[^\W\d_])[_]?(?P<index>\d{1,2})$")


def repeating_group_columns(column_names: Sequence[str]) -> set[str]:
    """
    Columns that belong to a numbered repeating group (``phone1``, ``phone2``, ...).

    The other 1NF violation, and the one no value statistic can see: there is no separator
    anywhere, the values are perfectly atomic, and only the *names* betray that a 1:n
    relationship was flattened sideways.

    Guard against the obvious false positive: if every column of the table shares one
    stem, that is a naming scheme (``col_01`` .. ``col_12``), not a repeating group.
    """
    stems: dict[str, set[str]] = {}
    for name in column_names:
        match = _REPEATING_SUFFIX.match(str(name))
        if match:
            stems.setdefault(match.group("stem").lower(), set()).add(name)

    grouped = {name for members in stems.values() if len(members) >= 2 for name in members}
    if len(grouped) == len(column_names):
        return set()
    return grouped


# --------------------------------------------------------------------------------------
# SQL helpers
# --------------------------------------------------------------------------------------

# NULL is treated as a value of its own, in both the single and the pairwise counts, so
# the two are comparable. CHR(30) and CHR(31) are non-printable and cannot occur in the
# data, which keeps `a="x|y", b="z"` from colliding with `a="x", b="y|z"`.
_NULL_SENTINEL = "CHR(30)"
_FIELD_SEPARATOR = "CHR(31)"


def _key_expr(column: str) -> str:
    return f'COALESCE(CAST("{column}" AS VARCHAR), {_NULL_SENTINEL})'


def _combination_expr(columns: Iterable[str]) -> str:
    """``count(DISTINCT ...)`` over one column or over a combination of them."""
    keys = [_key_expr(column) for column in columns]
    if len(keys) == 1:
        # Trino's CONCAT needs at least two arguments.
        return f"count(DISTINCT {keys[0]})"
    parts = f", {_FIELD_SEPARATOR}, ".join(keys)
    return f"count(DISTINCT CONCAT({parts}))"


def _is_text_type(data_type: str) -> bool:
    return data_type.lower().startswith(("varchar", "char"))


# --------------------------------------------------------------------------------------
# Cost budget (a): row sampling, (b): bounded search
# --------------------------------------------------------------------------------------

# Above this many rows the profile is measured on a sample instead of the whole table.
# Deliberately low enough that part of the *training* set is sampled too: the 50k-row
# tables cross it, the 500- and 5.000-row ones do not. A threshold that only ever fires in
# production would hand the model exact features while learning and sampled ones while
# predicting - finding 1.6 a second time, just harder to see.
SAMPLE_ROW_THRESHOLD = 20_000
SAMPLE_TARGET_ROWS = 10_000

# Pairwise measurement is `C(n,2)` aggregates - 780 at 40 columns, 19.900 at 200. It runs
# before the UCC search, so it is the one unbounded cost in the whole profile. Prose
# columns are dropped first (they can be neither key part nor a useful determinant), then
# the remainder is cut by ordinal position.
MAX_PROFILED_COLUMNS = 60


def count_rows(conn: Connection, qualified: str) -> int:
    """The table's true row count, before any sampling."""
    return int(conn.execute(text(f"SELECT count(*) FROM {qualified}")).scalar_one())  # noqa: S608 - identifiers come from information_schema


def _row_hash_expr(names: Sequence[str]) -> str:
    """A stable 64-bit hash of the whole row - the same value on every run and query."""
    keys = [_key_expr(name) for name in names]
    separator = f", {_FIELD_SEPARATOR}, "
    concatenated = keys[0] if len(keys) == 1 else f"CONCAT({separator.join(keys)})"
    return f"from_big_endian_64(xxhash64(to_utf8({concatenated})))"


def _sampled_source(
    qualified: str, names: Sequence[str], total_rows: int
) -> tuple[str, int | None]:
    """
    The FROM clause every aggregate runs against, and the sampling modulus that produced it.

    Not ``TABLESAMPLE BERNOULLI``: Trino takes no seed, so it draws different rows on every
    call. That breaks two things at once. The harness re-runs the build and expects the same
    numbers, and - far worse - the profile issues *many* queries. If each one saw a
    different sample, ``count(DISTINCT a)`` and ``count(DISTINCT a, b)`` would come from
    different row sets, and the test ``distinct(a, b) == distinct(a)`` that every dependency
    rests on would compare two unrelated numbers. A pair count could even come out below its
    own single count. The result would not be an approximate measurement but an incoherent
    one.

    A hash of the row against a fixed modulus has neither problem: deterministic across runs
    and identical in every query, so the sample behaves exactly like a smaller table.

    The error it introduces is one-sided, which is what makes it usable as a feature. An FD
    that holds on the table holds on every subset of it, and a combination unique on the
    table is unique on every subset - so nothing true is lost. What a sample can do is miss
    the one colliding row pair that refutes a candidate, making it *look* unique. False
    positives only, never false negatives.
    """
    if total_rows <= SAMPLE_ROW_THRESHOLD or not names:
        return qualified, None
    modulus = -(-total_rows // SAMPLE_TARGET_ROWS)  # ceil, without importing math
    hashed = _row_hash_expr(names)
    # mod() keeps the sign of the dividend in Trino, and abs() overflows on the minimum
    # bigint - the double mod is the safe way to land in [0, modulus).
    gate = f"mod(mod({hashed}, {modulus}) + {modulus}, {modulus}) = 0"
    return f"(SELECT * FROM {qualified} WHERE {gate})", modulus  # noqa: S608 - identifiers come from information_schema


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    data_type: str


def read_columns(conn: Connection, catalog: str, schema: str, table: str) -> list[ColumnInfo]:
    """Column names and types, in ordinal position order."""
    rows = conn.execute(
        # The catalog is an identifier and cannot be a bound parameter; schema and table
        # are bound below.
        text(f"""
            SELECT column_name, data_type
            FROM {catalog}.information_schema.columns
            WHERE table_schema = :schema AND table_name = :table
            ORDER BY ordinal_position
        """),  # noqa: S608 - catalog is an identifier, schema/table are bound
        {"schema": schema, "table": table},
    ).fetchall()
    return [ColumnInfo(name=row[0], data_type=row[1]) for row in rows]


# --------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TableProfile:
    """Everything measured on one table, before it is turned into features."""

    row_count: int
    columns: tuple[ColumnInfo, ...]
    distinct: dict[str, int]
    non_null: dict[str, int]
    list_like_ratio: dict[str, float]
    mean_length: dict[str, float]
    mean_separators: dict[str, float]
    pair_distinct: dict[frozenset[str], int]

    # Cost budget. `row_count` above is always the number of rows actually measured; these
    # say what that number is relative to the table it came from.
    source: str = ""  # the FROM clause used - a sampling subquery, or "" for the table
    sampled_from: int | None = None  # true row count when sampled, None when not

    # Single-column cardinalities over the *whole* table, even when everything else was
    # sampled. Empty when no sampling happened, in which case `distinct` already is it.
    #
    # Measured separately because sampling would otherwise break the one prune that makes
    # a level-3 search affordable. That prune asks whether the distinct counts multiply out
    # to the row count, so its strength comes entirely from the row count being large.
    # Measured on a fifth of a 50.000-row table, the threshold drops to 10.000, candidates
    # that could never be unique in the full table start qualifying, and the real key
    # drowns: 2.331 candidates became 6.986 and the true key fell from inside the tested
    # 400 to rank 4.679. Cardinalities cost n aggregates against the C(n,2) of the pairwise
    # matrix, so keeping them exact is close to free - it is the combinatorial half that
    # needed sampling in the first place.
    distinct_full: dict[str, int] = field(default_factory=dict)

    @property
    def total_rows(self) -> int:
        """Rows in the table, sampled or not - what a caller means by "how big is this"."""
        return self.sampled_from if self.sampled_from is not None else self.row_count

    def cardinality(self, column: str) -> int:
        """
        Distinct values of ``column`` in the whole table.

        Use this wherever a number is compared against ``total_rows`` - the prune, the
        ranking, the level-1 uniqueness test. Use ``distinct`` instead wherever a number is
        compared against a *pairwise* count, because those were measured on the sample and
        mixing the two would compare a fifth of a table against all of it.
        """
        return self.distinct_full.get(column) or self.distinct.get(column, 1)

    @property
    def pair_coverage(self) -> float:
        """Share of column pairs actually measured; below 1.0 the pair search was cut."""
        possible = len(self.columns) * (len(self.columns) - 1) // 2
        return len(self.pair_distinct) / possible if possible else 1.0


def profile_table(
    conn: Connection,
    catalog: str,
    schema: str,
    table: str,
    batch_size: int = 40,
) -> TableProfile:
    """
    One pass for the per-column statistics, one batched pass for the pairs.

    Args:
        conn: Open SQLAlchemy connection to Trino.
        catalog: Catalog name ("iceberg").
        schema: Schema holding the table.
        table: Table name.
        batch_size: Aggregates per query. Trino handles wide SELECT lists well, but a
            single query with thousands of ``count(DISTINCT ...)`` is where it stops
            being reasonable.
    """
    columns = read_columns(conn, catalog, schema, table)
    qualified = f"{catalog}.{schema}.{table}"
    names = [column.name for column in columns]

    total_rows = count_rows(conn, qualified)
    source, modulus = _sampled_source(qualified, names, total_rows)
    if modulus is not None:
        print(
            f"  {qualified}: sampling 1 row in {modulus} of {total_rows}, "
            f"targeting ~{SAMPLE_TARGET_ROWS}",
        )

    aggregates = ["count(*)"]
    for column in columns:
        aggregates.append(f"{_combination_expr([column.name])}")
        aggregates.append(f'count_if("{column.name}" IS NOT NULL)')
        if _is_text_type(column.data_type):
            quoted = f'"{column.name}"'
            aggregates += [
                f"count_if(regexp_like({quoted}, '{LIST_VALUE_PATTERN}'))",
                f"avg(CAST(length({quoted}) AS DOUBLE))",
                (
                    f"avg(CAST(length({quoted}) "
                    f"- length(regexp_replace({quoted}, '[,;|]', '')) AS DOUBLE))"
                ),
            ]

    # Strict on purpose: this pass is one cheap query and the cursor arithmetic below
    # depends on its length. A tolerated failure here would shift every later column's
    # statistics onto the wrong name. Only the pair pass, which is the expensive one and
    # whose results are a dict, is allowed to come back incomplete.
    values = _run_aggregates(conn, source, aggregates, batch_size)
    row_count = int(values[0])
    cursor = 1

    distinct: dict[str, int] = {}
    non_null: dict[str, int] = {}
    list_like_ratio: dict[str, float] = {}
    mean_length: dict[str, float] = {}
    mean_separators: dict[str, float] = {}

    for column in columns:
        distinct[column.name] = int(values[cursor] or 0)
        non_null[column.name] = int(values[cursor + 1] or 0)
        cursor += 2
        if _is_text_type(column.data_type):
            matches = int(values[cursor] or 0)
            present = non_null[column.name]
            list_like_ratio[column.name] = matches / present if present else 0.0
            mean_length[column.name] = float(values[cursor + 1] or 0.0)
            mean_separators[column.name] = float(values[cursor + 2] or 0.0)
            cursor += 3
        else:
            list_like_ratio[column.name] = 0.0
            mean_length[column.name] = 0.0
            mean_separators[column.name] = 0.0

    def token_length(name: str) -> float:
        tokens = mean_separators[name] + 1
        return mean_length[name] / tokens if tokens else 0.0

    pair_names = names
    if len(names) > MAX_PROFILED_COLUMNS:
        # Prose first: a comment column cannot be part of a key and any dependency into or
        # out of it is noise, so it is the cheapest thing to give up. Whatever is still
        # over the cap is cut by ordinal position - deliberately not by cardinality, which
        # would drop exactly the low-cardinality columns that FD determinants are made of.
        eligible = [name for name in names if token_length(name) <= PROSE_MAX_TOKEN_LENGTH]
        pair_names = (eligible or names)[:MAX_PROFILED_COLUMNS]
        print(
            f"  {qualified}: pair search capped at {len(pair_names)} of {len(names)} "
            f"columns, prose dropped first",
        )

    pair_distinct = _measure_pairs(conn, source, pair_names, row_count, batch_size)

    # n aggregates over the full table, against the C(n,2) of the pairwise matrix above -
    # see TableProfile.distinct_full for why sampling these would cost far more than it saves.
    distinct_full: dict[str, int] = {}
    if modulus is not None:
        full_values = _run_aggregates(
            conn,
            qualified,
            [_combination_expr([name]) for name in names],
            batch_size,
        )
        distinct_full = {
            name: int(value) for name, value in zip(names, full_values, strict=True) if value
        }

    return TableProfile(
        row_count=row_count,
        columns=tuple(columns),
        distinct=distinct,
        non_null=non_null,
        list_like_ratio=list_like_ratio,
        mean_length=mean_length,
        mean_separators=mean_separators,
        pair_distinct=pair_distinct,
        source=source,
        sampled_from=total_rows if modulus is not None else None,
        distinct_full=distinct_full,
    )


def _run_aggregates(
    conn: Connection,
    qualified: str,
    aggregates: Sequence[str],
    batch_size: int,
    tolerate_errors: bool = False,
) -> list:
    """
    Evaluate many aggregates over one table, in batches, preserving their order.

    With ``tolerate_errors`` a batch that fails - a Trino timeout on a wide table is the
    case this exists for - yields ``None`` for its measurements instead of losing the whole
    table. ``None`` means *unknown* and is dropped by the callers, never coerced to zero: a
    zero distinct count would read as "constant column" and quietly invent a dependency.
    """
    values: list = []
    for start in range(0, len(aggregates), batch_size):
        chunk = aggregates[start : start + batch_size]
        try:
            row = conn.execute(text(f"SELECT {', '.join(chunk)} FROM {qualified}")).fetchall()[0]  # noqa: S608 - identifiers come from information_schema
        except SQLAlchemyError as error:
            if not tolerate_errors:
                raise
            print(
                f"  {qualified}: aggregate batch failed, {len(chunk)} measurements "
                f"unknown ({type(error).__name__})",
            )
            values.extend([None] * len(chunk))
            continue
        values.extend(row)
    return values


def count_distinct_combinations(
    conn: Connection,
    qualified: str,
    combos: Sequence[frozenset[str]],
    batch_size: int = 40,
    tolerate_errors: bool = False,
) -> dict[frozenset[str], int]:
    """
    ``count(DISTINCT (...))`` for each attribute combination, batched into few queries.

    The primitive both the feature extraction and the validation harness (2c) are built
    on. A combination count answers three questions with one number: it is a key when the
    count equals the row count, ``X -> A`` holds when ``count(X u {A}) == count(X)``, and
    the ratio to the row count is the dependency's strength.
    """
    if not combos:
        return {}

    aggregates = [_combination_expr(sorted(combo)) for combo in combos]
    values = _run_aggregates(conn, qualified, aggregates, batch_size, tolerate_errors)
    # A combination whose batch failed is simply absent - see _run_aggregates on why it is
    # not a zero. Callers read this dict with .get() or iterate it, never by index.
    return {
        combo: int(value) for combo, value in zip(combos, values, strict=True) if value is not None
    }


def _measure_pairs(
    conn: Connection,
    qualified: str,
    names: Sequence[str],
    row_count: int,
    batch_size: int,
) -> dict[frozenset[str], int]:
    """Every column pair - see ``count_distinct_combinations`` for what the number means."""
    pairs = [frozenset(pair) for pair in combinations(names, 2)]
    if not pairs or row_count == 0:
        return {}
    return count_distinct_combinations(conn, qualified, pairs, batch_size, tolerate_errors=True)


# --------------------------------------------------------------------------------------
# Key search (decision E1) and dependency derivation
# --------------------------------------------------------------------------------------

MAX_UCC_LEVEL = 3

# A column whose pieces average more than this many characters is prose, and prose is not
# a key. Measured on values rather than on names on purpose: the name-based free-text veto
# cannot see an obfuscated column, and half the generated tables carry obfuscated names.
PROSE_MAX_TOKEN_LENGTH = 20

# A column that already separates this share of the rows does not need a partner - pairing
# it with anything else is almost guaranteed to be unique and says nothing about the
# schema. It stays eligible as a key on its own; only composites are filtered.
NEAR_UNIQUE_RATIO = 0.9

# Ceiling on the level-3 search. `C(40,3)` is 9880, and a wide table with a three-part
# key reaches level 3 with nearly all of them surviving the cardinality prune - a few
# hundred queries for a single table. Candidates are ranked before the cut, so what gets
# tested is the most plausible part of the space, and the cut is printed rather than
# swallowed.
MAX_LEVEL_3_CANDIDATES = 400


def ucc_score(profile: TableProfile, columns: Iterable[str]) -> float:
    """
    How *surprising* this combination's uniqueness is - the birthday estimate.

    Under independence, ``n`` rows drawn from ``prod(distinct)`` possible combinations
    collide about ``n^2 / (2 * prod)`` times. Observing zero collisions when many were
    expected means the uniqueness is structural; observing zero when none were expected
    means it came for free.

    Measured on the generated set: the real key ``{orderkey, linenumber}`` scores 568, an
    accidental pair like ``{shipdate, partkey}`` scores 1.2. That gap is what turns a flat
    set of 18-34 equally-valid UCCs into a ranking.

    Computed on the full table even when the counts were sampled: the score compares a row
    count against a product of cardinalities, and sampling shrinks the two by different
    factors. Scored on a fifth of the rows, the ranking inverts.
    """
    product = 1.0
    for column in columns:
        product *= max(profile.cardinality(column), 1)
    return (profile.total_rows**2) / (2 * product)


def non_atomic_columns(profile: TableProfile) -> set[str]:
    """
    Columns whose values are separator-lists of short tokens - the 1NF violation.

    Shared by the ``col_violates_1nf`` feature and by ``key_eligible_columns``, because the
    two have to agree by construction: a column cannot both violate 1NF and be a key part.
    """
    return {
        column.name
        for column in profile.columns
        if not is_freetext_name(column.name)
        and profile.list_like_ratio[column.name] >= LIST_VALUE_MIN_SHARE
        and _mean_token_length(profile, column.name) <= LIST_TOKEN_MAX_MEAN_LENGTH
    }


def key_eligible_columns(profile: TableProfile) -> set[str]:
    """
    Columns that could plausibly be part of a key at all.

    Two exclusions, both measured rather than argued:

    * **Prose.** A comment column is unique in almost every table, which makes it a
      perfectly valid UCC and a useless one. Left in, it outranks the real key in every
      table that has one - measured on 6 of 132.
    * **Non-atomic columns.** A column holding ``"AIR,RAIL,SHIP"`` per row is often unique
      and never a key: a key identifies a row, a flattened 1:n list is the thing 1NF
      forbids. Left in, `lineitem_shipmode_list` and `col_i_list` outranked the real key in
      5 of the 13 tables where the ranking failed - and worse, a 0NF table whose "key" is
      its list column gets its whole dependency classification built on that column.

    Near-unique columns are handled separately, in ``find_uccs``, because the rule differs
    by level: they are fine as a key on their own and accidental inside a composite.
    """
    non_atomic = non_atomic_columns(profile)
    return {
        column.name
        for column in profile.columns
        if _mean_token_length(profile, column.name) <= PROSE_MAX_TOKEN_LENGTH
        and column.name not in non_atomic
    }


def find_uccs(
    conn: Connection,
    qualified: str,
    profile: TableProfile,
    max_level: int = MAX_UCC_LEVEL,
    batch_size: int = 40,
    search: dict | None = None,
) -> tuple[list[frozenset[str]], bool]:
    """
    Minimal unique column combinations, level by level (decision **E1**).

    Levels 1 and 2 are free - they fall out of the counts already measured. Level 3 costs
    a query and only runs when the first two found nothing, pruned twice:

    * Apriori: a triple containing a smaller UCC is not minimal.
    * Cardinality: ``{a,b,c}`` can only be unique if the product of the distinct counts
      reaches the row count. Cheap arithmetic that removes most triples before any SQL.

    Every level is searched, not just the first one that yields something. Stopping at the
    first non-empty level was measured to be wrong: one accidentally unique column - a
    supply cost, a free-text comment - is a perfectly valid level-1 UCC, and stopping there
    hides the real composite key sitting at level 2. The partial dependency that defines
    2NF then becomes invisible, and every 1NF table looks like a 2NF one.

    Searching every level, though, produced the opposite problem: 18-34 UCCs per table
    where the real key is one pair. Two filters and a ranking bring that back down, each
    measured against the manifest's declared keys (which the extractor never sees):

    | step                                        | real key ranked first | UCCs/table |
    | :------------------------------------------ | --------------------: | ---------: |
    | flat set, no ranking                        |                     - |      18-34 |
    | + ranked by ``ucc_score``                   |               116/132 |            |
    | + prose columns not key-eligible            |               124/132 |            |
    | + near-unique columns out of composites     |           **128/132** |    **1-7** |

    Args:
        search: If given, filled with how much of the level-3 space was actually tested.
            "Found no key" and "stopped looking" are different statements, and without this
            they arrive at the model as the same number.

    Returns:
        The UCCs found, **best first**, and whether the search ran out of levels without
        finding one. The second value becomes ``table_has_no_ucc_le3`` and the review flag
        from **E1** - an honest "unknown" state has to exist in the training data, or the
        model learns that a key is always available.
    """
    if search is not None:
        search.update({"candidates": 0, "tested": 0, "truncated": False})

    rows = profile.row_count
    if rows == 0:
        return [], True

    source = profile.source or qualified
    # Only columns that got a pair measurement can take part in a composite: without their
    # pair counts neither the Apriori prune nor the level-2 result means anything for them.
    # Equal to every column unless the pair search was capped.
    measured = set().union(*profile.pair_distinct) if profile.pair_distinct else set()
    names = [
        column.name for column in profile.columns if column.name in key_eligible_columns(profile)
    ]
    # Both of these compare a cardinality against a row count, so both take the full-table
    # numbers. Levels 2 and 3 below compare against *pairwise* counts instead, which were
    # measured on the sample, and therefore use the sampled row count.
    total = profile.total_rows
    near_unique = {name for name in names if profile.cardinality(name) / total >= NEAR_UNIQUE_RATIO}

    def usable(combination: Iterable[str]) -> bool:
        """A composite may not lean on a column that is already nearly a key by itself."""
        return not near_unique.intersection(combination)

    uccs = [frozenset({name}) for name in names if profile.cardinality(name) == total]

    if max_level >= 2:
        uccs += [
            pair
            for pair, count in profile.pair_distinct.items()
            if count == rows
            and pair <= set(names)
            and usable(pair)
            and not any(ucc <= pair for ucc in uccs)
        ]

    if max_level >= 3:
        triple_names = [name for name in names if name in measured] if measured else names
        candidates = [
            frozenset(triple)
            for triple in combinations(triple_names, 3)
            if usable(triple)
            and _cardinality_allows(profile, triple, total)
            and not any(ucc <= frozenset(triple) for ucc in uccs)
        ]
        if search is not None:
            search["candidates"] = len(candidates)
            search["tested"] = min(len(candidates), MAX_LEVEL_3_CANDIDATES)
        if len(candidates) > MAX_LEVEL_3_CANDIDATES:
            if search is not None:
                search["truncated"] = True
            # A 40-column table with a three-part key reaches level 3 with ~10.000 triples
            # that all survive the cardinality prune, which is a few hundred queries for
            # one table. Testing the most plausible ones first is the trade; saying so out
            # loud is the part that matters, because a silent cap reads as "searched
            # everything" when it did not.
            candidates.sort(key=lambda triple: -ucc_score(profile, triple))
            print(
                f"  {qualified}: level-3 search capped at {MAX_LEVEL_3_CANDIDATES} of "
                f"{len(candidates)} candidates, highest-scoring first",
            )
            candidates = candidates[:MAX_LEVEL_3_CANDIDATES]
        if candidates:
            aggregates = [_combination_expr(sorted(triple)) for triple in candidates]
            values = _run_aggregates(conn, source, aggregates, batch_size, tolerate_errors=True)
            uccs += [
                triple
                for triple, count in zip(candidates, values, strict=True)
                if count is not None and int(count) == rows
            ]
            if search is not None and any(value is None for value in values):
                search["truncated"] = True
                search["tested"] -= sum(1 for value in values if value is None)

    # Best first. Everything downstream that has to pick *a* key - and the review queue,
    # when it has to show a human one line instead of thirty - takes the head of this list.
    #
    # Ties broken by ordinal position, not alphabetically. Two fully unique single columns
    # score identically by construction: on TPC-H `customer`, `custkey` and `address` are
    # both 1500 distinct over 1500 rows, and the alphabetical tiebreak picked `address` - a
    # free-text field - as the table's key. Position is a weak signal but a real one, and
    # unlike the alphabet it is not arbitrary: a key sits at the front of a table by
    # convention. Positions are unique, so the order is fully determined.
    uccs.sort(key=lambda ucc: (-ucc_score(profile, ucc), _positions(profile, ucc)))
    return uccs, not uccs


def _positions(profile: TableProfile, columns: Iterable[str]) -> list[int]:
    """Ordinal positions of ``columns``, ascending - the tiebreaker for equal-scoring keys."""
    order = {column.name: index for index, column in enumerate(profile.columns)}
    return sorted(order.get(name, len(order)) for name in columns)


def _cardinality_allows(profile: TableProfile, columns: Iterable[str], rows: int) -> bool:
    """
    Necessary condition for uniqueness: the distinct counts must multiply out to at least
    the row count. Pure arithmetic on numbers already measured, and it removes most
    triples before any SQL is issued.

    Both sides come from the full table, never from a sample - see
    ``TableProfile.distinct_full``. ``rows`` is passed in rather than read off the profile
    only so the caller stays explicit about which count it means.
    """
    product = 1
    for name in columns:
        product *= max(profile.cardinality(name), 1)
        if product >= rows:
            return True
    return False


# --------------------------------------------------------------------------------------
# Error-tolerant dependencies and keys (decision (c), cheap half)
# --------------------------------------------------------------------------------------

# A dependency this close to holding counts as *nearly* holding. Deliberately reported
# alongside the exact counts rather than replacing them: every threshold calibrated so far -
# the key ranking, the signature limit - was measured against exact dependencies, and
# loosening them in place would invalidate all of it at once.
NEAR_FD_MIN_STRENGTH = 0.95

# Same idea for keys: a combination that separates all but a handful of rows. On dirty data
# this is what a real primary key looks like once a few rows have been duplicated.
NEAR_UCC_TOLERANCE = 0.01


def dependency_strength(profile: TableProfile, determinant: str, dependent: str) -> float:
    """
    How close ``determinant -> dependent`` is to holding, between 0 and 1.

    ``distinct(X) / distinct(X u {A})`` - exactly 1.0 when the dependency holds, and falling
    as violations split groups apart. Free: both numbers are already in the pairwise matrix,
    so nothing extra is asked of the database.

    This is a *group-level* error, not the textbook ``g3``, and the difference matters. ``g3``
    is the share of rows one would have to delete, which needs the size of each group -
    ``sum_x max_a count(x, a)`` - and that is a nested GROUP BY per pair. Nested aggregates
    cannot be batched into a wide SELECT list the way ``count(DISTINCT ...)`` can, so exact
    ``g3`` would cost one query per pair, C(n,2) of them. One violated group of a thousand
    rows and a thousand violated groups of one row score the same here; separating those is
    what the expensive measure would buy.
    """
    both = profile.pair_distinct.get(frozenset({determinant, dependent}))
    groups = profile.distinct.get(determinant)
    if not both or not groups:
        return 0.0
    return min(groups / both, 1.0)


def find_near_dependencies(
    profile: TableProfile,
    min_strength: float = NEAR_FD_MIN_STRENGTH,
) -> list[tuple[str, str, float]]:
    """
    Dependencies that almost hold - strong enough to mean something, not exact.

    Exact ones are excluded: they are already counted by ``find_dependencies``, and a feature
    that mixed the two would say nothing the exact count does not.
    """
    rows = profile.row_count
    if not rows:
        return []
    constants = constant_columns(profile)
    found = []
    for pair in profile.pair_distinct:
        left, right = sorted(pair)
        for determinant, dependent in ((left, right), (right, left)):
            # The same three trivial cases find_dependencies drops - see it for the reasons.
            if dependent in constants:
                continue
            if profile.distinct.get(determinant, 0) / rows >= NEAR_UNIQUE_RATIO:
                continue
            strength = dependency_strength(profile, determinant, dependent)
            if min_strength <= strength < 1.0:
                found.append((determinant, dependent, strength))
    return sorted(found, key=lambda item: (-item[2], item[0], item[1]))


def find_near_uccs(
    profile: TableProfile,
    tolerance: float = NEAR_UCC_TOLERANCE,
) -> list[frozenset[str]]:
    """
    Combinations that separate all but ``tolerance`` of the rows, without being unique.

    Levels 1 and 2 only, and no SQL at all: both come out of counts already measured. A
    level-3 near-key would need its own search, which is the expensive half of (c) and waits
    for dirty data to calibrate against.

    Returns:
        Near-keys, best first, excluding anything that already contains an exact key - a
        near-key sitting on top of a real one says nothing new.
    """
    rows = profile.row_count
    if rows == 0:
        return []

    floor = rows * (1 - tolerance)
    eligible = key_eligible_columns(profile)
    near = [frozenset({name}) for name in eligible if floor <= profile.distinct.get(name, 0) < rows]
    near += [
        pair
        for pair, count in profile.pair_distinct.items()
        if floor <= count < rows and pair <= eligible and not any(single <= pair for single in near)
    ]
    near.sort(
        key=lambda combination: (
            -ucc_score(profile, combination),
            _positions(profile, combination),
        ),
    )
    return near


MIN_PAIR_FD_SUPPORT = 10
MAX_PAIR_FD_TRIPLES = 1500


def pair_fd_candidates(profile: TableProfile) -> list[tuple[frozenset[str], str]]:
    """
    The ``(determinant pair, dependent)`` triples worth measuring, best-evidenced first.

    Three prunes, all free - they run on numbers the pair matrix already holds:

    * **Support.** ``rows - distinct(pair)`` is how often the determinant repeats; below
      ``MIN_PAIR_FD_SUPPORT`` there is nothing for a dependency to hold *over*.
    * **Single-explained.** ``{a, b} -> c`` restates ``a -> c`` whenever the single holds,
      so those dependents are skipped. Judged on the raw single-attribute facts, before
      any guard: a pair FD is not news just because its single was dropped as trivial.
    * **Eligibility.** Prose and list columns cannot be determinant parts, and a constant
      dependent is determined by everything (see ``find_dependencies``).
    """
    rows = profile.row_count
    if not rows:
        return []
    names = [column.name for column in profile.columns]
    eligible = key_eligible_columns(profile)
    constants = constant_columns(profile)

    explained: set[tuple[str, str]] = set()
    for pair, both in profile.pair_distinct.items():
        left, right = sorted(pair)
        for determinant, dependent in ((left, right), (right, left)):
            if both == profile.distinct.get(determinant):
                explained.add((determinant, dependent))

    candidates: list[tuple[frozenset[str], str]] = []
    for pair, both in profile.pair_distinct.items():
        if not pair <= eligible:
            continue
        if rows - both < MIN_PAIR_FD_SUPPORT:
            continue
        left, right = sorted(pair)
        for dependent in names:
            if dependent in pair or dependent in constants:
                continue
            if (left, dependent) in explained or (right, dependent) in explained:
                continue
            # Necessary conditions, free from counts already measured: the triple count
            # is at least every projection's count, so ``distinct(a,b,c) == distinct(a,b)``
            # is impossible whenever the dependent alone - or either sub-pair - already
            # exceeds the determinant's count. This is what keeps a wide table's
            # candidate list from being C(n,2) * (n-2) queries.
            if profile.distinct.get(dependent, 0) > both:
                continue
            left_pair = profile.pair_distinct.get(frozenset({left, dependent}))
            right_pair = profile.pair_distinct.get(frozenset({right, dependent}))
            if (left_pair is not None and left_pair > both) or (
                right_pair is not None and right_pair > both
            ):
                continue
            candidates.append((pair, dependent))

    candidates.sort(
        key=lambda item: (
            -(rows - profile.pair_distinct[item[0]]),
            _positions(profile, item[0]),
            _positions(profile, [item[1]]),
        ),
    )
    return candidates


def find_pair_dependencies(
    conn: Connection,
    qualified: str,
    profile: TableProfile,
    batch_size: int = 40,
    search: dict | None = None,
) -> list[tuple[frozenset[str], str]]:
    """
    Exact dependencies ``{a, b} -> c`` whose determinant is a column pair.

    ``{a, b} -> c`` holds when ``distinct(a, b, c) == distinct(a, b)`` - the same test the
    single search uses, one level up. The pair counts are already in the profile; only the
    triple counts cost queries, measured on the same (possibly sampled) source as the pair
    matrix so the two sides of the comparison come from identical rows. Sampling keeps its
    one-sided error: a pair FD can be invented by a sample, never lost to one.

    Args:
        search: If given, filled with ``pair_candidates`` / ``pair_tested`` /
            ``pair_truncated`` - "found nothing" and "stopped looking" must stay
            distinguishable, exactly as with the UCC search.
    """
    if search is not None:
        search.update({"pair_candidates": 0, "pair_tested": 0, "pair_truncated": False})

    candidates = pair_fd_candidates(profile)
    if search is not None:
        search["pair_candidates"] = len(candidates)
        search["pair_tested"] = min(len(candidates), MAX_PAIR_FD_TRIPLES)
    if not candidates:
        return []

    if len(candidates) > MAX_PAIR_FD_TRIPLES:
        if search is not None:
            search["pair_truncated"] = True
        print(
            f"  {qualified}: pair-FD search capped at {MAX_PAIR_FD_TRIPLES} of "
            f"{len(candidates)} candidates, highest support first",
        )
        candidates = candidates[:MAX_PAIR_FD_TRIPLES]

    source = profile.source or qualified
    triples = sorted({pair | {dependent} for pair, dependent in candidates}, key=sorted)
    counts = count_distinct_combinations(conn, source, triples, batch_size, tolerate_errors=True)
    if search is not None and len(counts) < len(triples):
        search["pair_truncated"] = True
        search["pair_tested"] -= sum(
            1 for pair, dependent in candidates if (pair | {dependent}) not in counts
        )

    return [
        (pair, dependent)
        for pair, dependent in candidates
        if counts.get(pair | {dependent}) == profile.pair_distinct[pair]
    ]


@dataclass(frozen=True)
class PairDependencyClasses:
    """Discovered pair-determinant dependencies split by what they would violate."""

    partial: list[tuple[frozenset[str], str]]
    transitive: list[tuple[frozenset[str], str]]


def classify_pair_dependencies(
    dependencies: Sequence[tuple[frozenset[str], str]],
    uccs: Sequence[frozenset[str]],
) -> PairDependencyClasses:
    """
    Where a pair determinant sits relative to the discovered keys.

    * a proper subset of a candidate key -> **partial**, breaks 2NF
    * containing no key and inside none -> **transitive**, breaks 3NF
    * a superset of a key -> neither: a superkey determines everything, trivially

    Judged against ALL discovered keys, not the top-ranked one - see the anykey variants
    in ``build_features`` for why both judgements exist.
    """
    partial: list[tuple[frozenset[str], str]] = []
    transitive: list[tuple[frozenset[str], str]] = []
    for determinant, dependent in dependencies:
        if any(determinant < key for key in uccs):
            partial.append((determinant, dependent))
        elif not any(key <= determinant for key in uccs):
            transitive.append((determinant, dependent))
    return PairDependencyClasses(partial=partial, transitive=transitive)


def constant_columns(profile: TableProfile) -> set[str]:
    """
    Columns holding a single value across the whole table.

    Reported as a feature of their own rather than only being excluded below, because
    "this table has three columns that never vary" is real information about the data and
    dropping it silently would lose it.
    """
    return {name for name, count in profile.distinct.items() if count <= 1}


def find_dependencies(profile: TableProfile) -> list[tuple[str, str]]:
    """
    Single-attribute dependencies ``a -> b`` that hold in the data.

    ``a -> b`` holds when adding ``b`` does not split any group of ``a``, i.e. when
    ``distinct(a, b) == distinct(a)``.

    Three kinds of finding are dropped as trivial - they hold by arithmetic rather than by
    anything about the schema, and each was measured to distort a real table:

    * **Unique determinant.** A key determines everything. Always excluded.
    * **Constant dependent.** A column with one value is determined by *every* other column.
      On TPC-H `orders`, `shippriority` is 0 in all 15.000 rows, which produced seven
      dependencies - ``custkey -> shippriority``, ``orderstatus -> shippriority``, and so on
      for every column - and pushed a 3NF table to 2NF.
    * **Near-unique determinant.** With 1499 distinct values over 1500 rows there is almost
      no grouping left to violate a dependency, so one appears by chance. On TPC-H
      `customer` that gave ``acctbal -> mktsegment`` and again cost a normal form. The same
      ratio already keeps near-unique columns out of composite keys; a determinant is the
      same argument.

    All three were absent from the generated training set by construction: the recipes
    deliberately left constant and near-unique columns out of the base catalog because they
    polluted the labels, which removed exactly the cases that then broke on real data.
    """
    rows = profile.row_count
    if not rows:
        return []
    constants = constant_columns(profile)
    found = []
    for pair, both in profile.pair_distinct.items():
        left, right = sorted(pair)
        for determinant, dependent in ((left, right), (right, left)):
            if dependent in constants:
                continue  # everything determines a constant - not a finding
            if profile.distinct[determinant] / rows >= NEAR_UNIQUE_RATIO:
                continue  # a key, or near enough to one, determines everything
            if both == profile.distinct[determinant]:
                found.append((determinant, dependent))
    return found


# --------------------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------------------


# --------------------------------------------------------------------------------------
# The contract between training and serving
# --------------------------------------------------------------------------------------

# Columns that identify a row rather than describe it. Never features.
IDENTITY_COLUMNS: tuple[str, ...] = (
    "database",
    "schema",
    "table_name",
    "column_name",
    "column_type",
)

# Every feature ``build_features`` produces, in the order it produces them.
#
# Declared here rather than read off a frame, because four places need to agree on it: the
# training script's X, the Pydantic request schema, the Prefect pipeline's payload, and the
# MLflow signature. The old set was "kept in sync by hand" across three of those - which is
# the same failure mode as finding 1.6, one formula in two copies that agree until one is
# edited. ``test_the_feature_contract_matches_what_is_built`` fails if this drifts from
# what ``build_features`` returns.
#
# ``column_type`` is in IDENTITY_COLUMNS, not here: it is one-hot encoded before training,
# so the columns the model actually sees are ``column_type_*`` and depend on which types
# occur in the data. The serving schema appends those.
FEATURE_COLUMNS: tuple[str, ...] = (
    "ordinal_position",
    "col_unique_ratio",
    "col_null_ratio",
    "col_list_like_ratio",
    "col_mean_token_length",
    "col_mean_token_count",
    "col_separator_density",
    "col_is_freetext_name",
    "col_violates_1nf",
    "col_in_repeating_group",
    "col_in_candidate_key",
    "col_is_prime",
    "col_determines_count",
    "col_depends_on_count",
    "table_row_count",
    "table_sampled",
    "table_sample_ratio",
    "table_search_truncated",
    "table_pair_coverage",
    "table_ucc_search_coverage",
    "table_column_count",
    "table_max_list_like_ratio",
    "table_ratio_list_like_columns",
    "table_repeating_group_ratio",
    "table_constant_column_ratio",
    "table_has_no_ucc_le3",
    "table_candidate_key_count",
    "table_key_size",
    "table_composite_key_count",
    "table_prime_ratio",
    "table_fd_count",
    "table_fd_ratio",
    "table_partial_fd_count",
    "table_partial_fd_ratio",
    "table_transitive_fd_count",
    "table_transitive_fd_ratio",
    "table_strict_partial_fd_count",
    "table_strict_transitive_fd_count",
    "table_near_fd_count",
    "table_near_fd_ratio",
    "table_max_near_fd_strength",
    "table_near_ucc_count",
    "table_has_near_key_but_no_key",
    # The single-attribute counts above judge partial-vs-transitive against the TOP-RANKED
    # key only (see classify_dependencies for the nf_018 measurement behind that). These
    # judge against ALL discovered keys instead. Both are reported because each is wrong in
    # a different direction - top-1 misread Willibald's kunde (GueltigBis sits in the
    # third-ranked key, so its FD was counted transitive and the table came back 2NF
    # instead of 1NF), while any-key inflates on tables with many accidental UCCs. Which
    # judgement carries signal is the model's question, not this module's.
    "table_partial_fd_count_anykey",
    "table_transitive_fd_count_anykey",
    # Pair-determinant dependencies - see MIN_PAIR_FD_SUPPORT for the Willibald
    # measurement that forced these to exist. Classified against all keys: a top-1-only
    # judgement makes no sense for pairs, since a single-column top key can never contain
    # one.
    "table_pair_fd_count",
    "table_pair_partial_fd_count",
    "table_pair_transitive_fd_count",
    "table_pair_fd_search_coverage",
)

# Features the serving schema has to accept as float; everything else in FEATURE_COLUMNS is
# an int, and the `column_type_*` dummies are bool. Listed rather than derived from the name:
# `table_fd_count` and `col_mean_token_count` both end in "count" and only one of them is an
# integer, so any naming rule would need an exception list anyway.
FLOAT_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "col_unique_ratio",
        "col_null_ratio",
        "col_list_like_ratio",
        "col_mean_token_length",
        "col_mean_token_count",
        "col_separator_density",
        "table_sample_ratio",
        "table_pair_coverage",
        "table_ucc_search_coverage",
        "table_max_list_like_ratio",
        "table_ratio_list_like_columns",
        "table_repeating_group_ratio",
        "table_constant_column_ratio",
        "table_prime_ratio",
        "table_fd_ratio",
        "table_partial_fd_ratio",
        "table_transitive_fd_ratio",
        "table_near_fd_ratio",
        "table_max_near_fd_strength",
        "table_pair_fd_search_coverage",
    },
)


# The column types that occur in the training set, and the one that serves as the reference
# category. Encoded explicitly rather than with `pd.get_dummies(drop_first=True)`, which
# picks the reference by alphabetical accident - add a "char" column to the data once and
# every dummy silently shifts meaning.
#
# An unseen type (timestamp, decimal, char) lands on all-zero dummies and is therefore read
# as the reference. That is a real limitation, not a bug to fix here: the alternative is a
# feature the trained model has never seen. It is worth knowing when predicting on a source
# whose types differ from the training set's.
COLUMN_TYPE_REFERENCE = "bigint"
COLUMN_TYPE_CATEGORIES: tuple[str, ...] = ("bigint", "date", "double", "integer", "varchar")
COLUMN_TYPE_DUMMIES: tuple[str, ...] = tuple(
    f"column_type_{name}" for name in COLUMN_TYPE_CATEGORIES if name != COLUMN_TYPE_REFERENCE
)


def normalise_column_type(raw: str) -> str:
    """``varchar(25)`` -> ``varchar``, ``int`` -> ``integer``. Trino spells widths inline."""
    name = re.sub(r"\(.*\)", "", str(raw)).strip().lower()
    return "integer" if name == "int" else name


def encode_column_type(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Add the ``column_type_*`` dummies, in ``COLUMN_TYPE_DUMMIES`` order.

    Shared by the training script and the serving pipeline for the same reason the features
    themselves are: an encoding applied twice from two code paths is an encoding that will
    eventually disagree with itself.
    """
    frame = frame.copy()
    normalised = frame["column_type"].map(normalise_column_type)
    for dummy in COLUMN_TYPE_DUMMIES:
        frame[dummy] = (normalised == dummy[len("column_type_") :]).astype(int)
    return frame


def build_features(
    conn: Connection,
    catalog: str,
    schema: str,
    table: str,
    diagnostics: dict | None = None,
    batch_size: int = 40,
) -> pd.DataFrame:
    """
    One feature row per column of ``table``.

    Table-level features are repeated on every row: the model sees a single row per call
    and cannot aggregate anything itself - dropping table-level aggregates as "derivable"
    once cost 0.22 F1 (finding 1.1).

    Args:
        conn: Open SQLAlchemy connection to Trino.
        catalog: Catalog name ("iceberg").
        schema: Schema holding the table.
        table: Table name.
        diagnostics: If given, the discovered keys and dependencies are written into it.
            The validation harness (2c) needs them to ask whether the *discovered*
            dependencies would change the label - and getting them out of here costs
            nothing, whereas re-profiling every table to recover them doubles the run.
        batch_size: Aggregates per query - see ``profile_table``.

    Returns:
        A frame whose columns are exactly ``IDENTITY_COLUMNS + FEATURE_COLUMNS``.
    """
    qualified = f"{catalog}.{schema}.{table}"
    profile = profile_table(conn, catalog, schema, table, batch_size=batch_size)
    names = [column.name for column in profile.columns]
    rows = profile.row_count

    search: dict = {}
    uccs, no_key_found = find_uccs(conn, qualified, profile, batch_size=batch_size, search=search)
    ucc_coverage = search["tested"] / search["candidates"] if search.get("candidates") else 1.0
    dependencies = find_dependencies(profile)
    pair_dependencies = find_pair_dependencies(
        conn,
        qualified,
        profile,
        batch_size=batch_size,
        search=search,
    )
    pair_fd_coverage = (
        search["pair_tested"] / search["pair_candidates"] if search.get("pair_candidates") else 1.0
    )

    # The best-ranked UCC is treated as *the* key, and prime means "part of it".
    #
    # Not the union of every UCC, which is what this was. Decision E2 introduced the ranking
    # precisely because a flat set of accidental UCCs is useless - but the classification
    # below kept consuming the flat set, so E2 only ever fixed half the problem. Measured
    # consequence on nf_018: 17 candidate keys, so `any(determinant in key for key in uccs)`
    # was true for nearly every determinant, `table_partial_fd_count` came out at 9 against a
    # 1NF median of 1, `table_transitive_fd_count` collapsed to 0, and `table_prime_ratio`
    # sat at 0.69 against a median of 0.3. The table was read as 2NF.
    #
    # How many UCCs there were is not lost - `table_candidate_key_count` and
    # `table_composite_key_count` still report it. Only the *position* judgement now rests on
    # the one key the ranking considers most plausible.
    primary_key = uccs[0] if uccs else frozenset()
    prime = primary_key

    near_dependencies = find_near_dependencies(profile)
    near_uccs = find_near_uccs(profile)

    repeating = repeating_group_columns(names)
    freetext = {name for name in names if is_freetext_name(name)}
    violating = non_atomic_columns(profile)
    constants = constant_columns(profile)

    classes = classify_dependencies(dependencies, [primary_key] if primary_key else [], prime)
    # The same dependencies judged against every discovered key instead of the best-ranked
    # one, plus the pair-determinant findings - see FEATURE_COLUMNS for why both exist.
    all_prime = frozenset().union(*uccs) if uccs else frozenset()
    classes_anykey = classify_dependencies(dependencies, uccs, all_prime)
    pair_classes = classify_pair_dependencies(pair_dependencies, uccs)
    # Keys whose uniqueness is SURPRISING (>= 1 expected collision under independence,
    # and none observed - the birthday estimate the ranking already uses). kunde's
    # {gueltigbis, mobil, telefon} is unique over 300 rows at 0.014 expected collisions:
    # arithmetic, not structure. Only these keys decide prime-ness in the rule label,
    # otherwise every accidental composite turns a transitive dependency into a
    # "partial" one and demotes the table a form too far.
    #
    # KNOWN LIMITATION: this filter degrades to no-op on very small tables. For any single
    # column that happens to be fully unique, ucc_score is rows/2 - so it clears the 1.0
    # bar for every table with >= 2 rows, unique or not. Measured on Chinook's real
    # employee table (8 rows, not in training): lastname/firstname/address/postalcode/
    # fax/email are each "structural" keys on that basis alone, {phone, reportsto} clears
    # the bar too (1.14 expected collisions), and the rule label reads {phone} -> city and
    # {reportsto} -> city off them as partial dependencies - a phone number does not
    # determine an office city by any design, it is 8 people's data agreeing by chance.
    # The row-count floor on the *reviewer* side catches this in practice (see
    # LOW_EVIDENCE_MIN_ROWS in normalform_pipeline.py: 8 rows is far under 30), but the
    # score itself carries no lower bound on rows and will keep making the same mistake
    # wherever nobody is checking the row count. Tightening ucc_score to also require a
    # minimum row count is the fix, not yet done - kunde's 300-row case is why the >= 1.0
    # bar exists at all; Chinook's 8-row case is why it is not sufficient by itself.
    structural_uccs = [key for key in uccs if ucc_score(profile, key) >= 1.0]
    if diagnostics is not None:
        diagnostics.update(
            {
                "columns": names,
                "row_count": rows,
                "total_row_count": profile.total_rows,
                "sampled": profile.sampled_from is not None,
                "pair_coverage": profile.pair_coverage,
                "ucc_search_coverage": ucc_coverage,
                "uccs": [sorted(key) for key in uccs],
                "structural_uccs": [sorted(key) for key in structural_uccs],
                "primary_key": sorted(primary_key),
                "no_key_found": no_key_found,
                "dependencies": [list(pair) for pair in dependencies],
                "pair_dependencies": [
                    [sorted(determinant), dependent] for determinant, dependent in pair_dependencies
                ],
                "constant_columns": sorted(constants),
                "near_dependencies": [
                    [determinant, dependent, round(strength, 4)]
                    for determinant, dependent, strength in near_dependencies
                ],
                "near_uccs": [sorted(combination) for combination in near_uccs],
                "violating_columns": sorted(violating),
                "repeating_columns": sorted(repeating),
            },
        )

    determines = dict.fromkeys(names, 0)
    depends_on = dict.fromkeys(names, 0)
    for determinant, dependent in dependencies:
        determines[determinant] += 1
        depends_on[dependent] += 1

    column_count = len(names) or 1
    possible_pairs = column_count * (column_count - 1) or 1
    table_level = {
        # The true row count, not the sampled one: how big the table is stays a fact about
        # the table. How much of it was measured is `table_sample_ratio`, one line below.
        "table_row_count": profile.total_rows,
        # Cost budget. Without these the model cannot tell a table that has no key from one
        # where the search was cut short - both arrive as table_has_no_ucc_le3 = 1.
        "table_sampled": int(profile.sampled_from is not None),
        "table_sample_ratio": rows / profile.total_rows if profile.total_rows else 1.0,
        "table_search_truncated": int(
            profile.pair_coverage < 1.0
            or search.get("truncated", False)
            or search.get("pair_truncated", False),
        ),
        "table_pair_coverage": profile.pair_coverage,
        "table_ucc_search_coverage": ucc_coverage,
        "table_column_count": len(names),
        "table_max_list_like_ratio": max(profile.list_like_ratio.values(), default=0.0),
        "table_ratio_list_like_columns": len(violating) / column_count,
        "table_repeating_group_ratio": len(repeating) / column_count,
        # Excluded from the dependency search as trivial - see find_dependencies - but told
        # to the model, because "three columns never vary" describes the data rather than
        # being noise to hide. On TPC-H `orders` this is 1 of 9.
        "table_constant_column_ratio": len(constants) / column_count,
        "table_has_no_ucc_le3": int(no_key_found),
        "table_candidate_key_count": len(uccs),
        # The size of the best-ranked key, not the smallest of all of them: an accidentally
        # unique column would otherwise report a key size of 1 for a table whose real key is
        # a pair, which is the same confusion E2 removed from the ranking.
        "table_key_size": len(primary_key),
        # Reported separately from the minimum, because an accidentally unique column
        # (a price, a free-text comment) hides a real composite key behind a minimum of 1.
        # A model that only saw the minimum could not tell the two situations apart.
        "table_composite_key_count": sum(1 for key in uccs if len(key) > 1),
        "table_prime_ratio": len(prime) / column_count,
        "table_fd_count": len(dependencies),
        "table_fd_ratio": len(dependencies) / possible_pairs,
        "table_partial_fd_count": len(classes.partial),
        "table_partial_fd_ratio": len(classes.partial) / possible_pairs,
        "table_transitive_fd_count": len(classes.transitive),
        "table_transitive_fd_ratio": len(classes.transitive) / possible_pairs,
        # Textbook variant, with the prime condition on the dependent. Kept so the two
        # can be compared on real data instead of argued about - see classify_dependencies.
        "table_strict_partial_fd_count": len(classes.strict_partial),
        "table_strict_transitive_fd_count": len(classes.strict_transitive),
        # Error-tolerant half of (c). Reported next to the exact counts, never instead of
        # them - on clean data these stay near zero, and they are what will carry the signal
        # once the dirty-data axis exists. Cost: nothing, both come out of the pair matrix.
        "table_near_fd_count": len(near_dependencies),
        "table_near_fd_ratio": len(near_dependencies) / possible_pairs,
        "table_max_near_fd_strength": max(
            (strength for *_, strength in near_dependencies),
            default=0.0,
        ),
        "table_near_ucc_count": len(near_uccs),
        "table_has_near_key_but_no_key": int(bool(near_uccs) and no_key_found),
        # Judged against all keys instead of the top-ranked one - see FEATURE_COLUMNS.
        "table_partial_fd_count_anykey": len(classes_anykey.partial),
        "table_transitive_fd_count_anykey": len(classes_anykey.transitive),
        # Pair-determinant findings; the Willibald misses live here.
        "table_pair_fd_count": len(pair_dependencies),
        "table_pair_partial_fd_count": len(pair_classes.partial),
        "table_pair_transitive_fd_count": len(pair_classes.transitive),
        "table_pair_fd_search_coverage": pair_fd_coverage,
    }

    records = []
    for position, column in enumerate(profile.columns, start=1):
        name = column.name
        records.append(
            {
                "database": catalog,
                "schema": schema,
                "table_name": table,
                "column_name": name,
                "column_type": column.data_type,
                "ordinal_position": position,
                # Full-table cardinality over full-table rows. The sampled ratio would be
                # biased upward - a fifth of the rows cannot hold more than a fifth of the
                # values, so every column looks more unique than it is.
                "col_unique_ratio": (
                    profile.cardinality(name) / profile.total_rows if profile.total_rows else 0.0
                ),
                "col_null_ratio": 1 - (profile.non_null[name] / rows) if rows else 0.0,
                "col_list_like_ratio": profile.list_like_ratio[name],
                "col_mean_token_length": _mean_token_length(profile, name),
                "col_mean_token_count": profile.mean_separators[name] + 1,
                "col_separator_density": (
                    profile.mean_separators[name] / profile.mean_length[name]
                    if profile.mean_length[name]
                    else 0.0
                ),
                "col_is_freetext_name": int(name in freetext),
                "col_violates_1nf": int(name in violating),
                "col_in_repeating_group": int(name in repeating),
                "col_in_candidate_key": int(any(name in key for key in uccs)),
                "col_is_prime": int(name in prime),
                "col_determines_count": determines[name],
                "col_depends_on_count": depends_on[name],
                **table_level,
            },
        )
    return pd.DataFrame(records)


def _mean_token_length(profile: TableProfile, column: str) -> float:
    """Mean length of the pieces between separators - what tells a list from prose."""
    tokens = profile.mean_separators[column] + 1
    return profile.mean_length[column] / tokens if tokens else 0.0


@dataclass(frozen=True)
class DependencyClasses:
    """Discovered dependencies split by what they would violate."""

    partial: list[tuple[str, str]]
    transitive: list[tuple[str, str]]
    strict_partial: list[tuple[str, str]]
    strict_transitive: list[tuple[str, str]]


def classify_dependencies(
    dependencies: Sequence[tuple[str, str]],
    uccs: Sequence[frozenset[str]],
    prime: frozenset[str],
) -> DependencyClasses:
    """
    Split the discovered dependencies into the ones that break 2NF and the ones that
    break 3NF - the distinction no feature made before, and the gap the generator
    artifact in finding 1.3 filled.

    What separates them is where the determinant sits:

    * a proper part of a composite key -> **partial**, breaks 2NF
    * outside every key -> **transitive**, breaks 3NF

    Two variants are returned, because the textbook definition does not survive contact
    with discovered keys. It also requires the *dependent* to be non-prime, which is
    correct when the candidate keys are known - and useless when they are discovered: a
    near-unique column (a price, a free-text comment) forms a valid UCC with almost every
    other column, so the union of all UCCs covers the whole table and every attribute
    comes out prime. Measured on the generated set, that filter drove both strict counts
    to zero for every table, at 2000 rows and at 20000.

    The relaxed variant drops the prime condition on the dependent and keeps only the
    determinant's position. It is noisier - an accidental composite UCC can make a
    harmless dependency look partial - but it is not identically zero. Both are reported;
    which one carries signal is a question for the model, not for this function.
    """
    partial: list[tuple[str, str]] = []
    transitive: list[tuple[str, str]] = []
    strict_partial: list[tuple[str, str]] = []
    strict_transitive: list[tuple[str, str]] = []

    for determinant, dependent in dependencies:
        in_composite_key = any(determinant in key and len(key) > 1 for key in uccs)
        outside_every_key = all(determinant not in key for key in uccs)

        if in_composite_key:
            partial.append((determinant, dependent))
            if dependent not in prime:
                strict_partial.append((determinant, dependent))
        elif outside_every_key:
            transitive.append((determinant, dependent))
            if dependent not in prime:
                strict_transitive.append((determinant, dependent))

    return DependencyClasses(
        partial=partial,
        transitive=transitive,
        strict_partial=strict_partial,
        strict_transitive=strict_transitive,
    )


def derive_rule_normal_form(diagnostics: dict) -> tuple[int, str]:
    """
    The normal form the discovered structure implies, decided by rule rather than model.

    Consumed by the serving pipeline as a cross-check stored NEXT TO the model's
    prediction - never as a feature. It exists because the Willibald run showed the model
    being confidently wrong (0.97+ on six mispredicted tables): the model is a function of
    the features, so when the two disagree, either the features missed something or the
    model did - and both cases belong in front of a human rather than silently in
    nf_results.

    Judgements are the textbook ones with one refinement: prime-ness is decided over the
    STRUCTURAL candidate keys (``structural_uccs`` - uniqueness with at least one
    expected collision behind it, see ``ucc_score``), not over every accidental UCC a
    finite table happens to satisfy. kunde_periode_1 is the measured case: the trio
    {gueltigbis, mobil, telefon} is unique over 300 rows at 0.014 expected collisions,
    and counting it as a key turned the real transitive dependency
    gueltigbis -> kkfirma into a "partial" one - one normal form too low.
    ``diagnostics`` is the dict ``build_features`` fills - the discovered structure
    comes out of the same pass that built the features, so this costs no extra query.

    Returns:
        ``(normal_form, evidence)`` - the class 0..3 and the finding that decided it.
    """
    key_evidence = diagnostics.get("structural_uccs", diagnostics.get("uccs", []))
    uccs = [frozenset(key) for key in key_evidence]
    violating = diagnostics.get("violating_columns", [])
    repeating = diagnostics.get("repeating_columns", [])
    if violating or repeating:
        evidence = ", ".join(sorted({*violating, *repeating}))
        return 0, f"non-atomic values or repeating group: {evidence}"

    singles = [tuple(pair) for pair in diagnostics.get("dependencies", [])]
    all_prime = frozenset().union(*uccs) if uccs else frozenset()
    classes = classify_dependencies(singles, uccs, all_prime)
    pairs = [
        (frozenset(determinant), dependent)
        for determinant, dependent in diagnostics.get("pair_dependencies", [])
    ]
    pair_classes = classify_pair_dependencies(pairs, uccs)

    def render(determinant: str | frozenset[str], dependent: str) -> str:
        names = [determinant] if isinstance(determinant, str) else sorted(determinant)
        return f"{{{', '.join(names)}}} -> {dependent}"

    partial = [render(d, a) for d, a in classes.partial]
    partial += [render(d, a) for d, a in pair_classes.partial]
    if partial:
        return 1, f"partial dependency: {'; '.join(partial)}"

    transitive = [render(d, a) for d, a in classes.transitive]
    transitive += [render(d, a) for d, a in pair_classes.transitive]
    if transitive:
        return 2, f"transitive dependency: {'; '.join(transitive)}"

    return 3, "no partial and no transitive dependency discovered"
