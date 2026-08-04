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
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sqlalchemy import text


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

    values = _run_aggregates(conn, qualified, aggregates, batch_size)
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

    pair_distinct = _measure_pairs(conn, qualified, names, row_count, batch_size)

    return TableProfile(
        row_count=row_count,
        columns=tuple(columns),
        distinct=distinct,
        non_null=non_null,
        list_like_ratio=list_like_ratio,
        mean_length=mean_length,
        mean_separators=mean_separators,
        pair_distinct=pair_distinct,
    )


def _run_aggregates(
    conn: Connection,
    qualified: str,
    aggregates: Sequence[str],
    batch_size: int,
) -> list:
    """Evaluate many aggregates over one table, in batches, preserving their order."""
    values: list = []
    for start in range(0, len(aggregates), batch_size):
        chunk = aggregates[start : start + batch_size]
        row = conn.execute(text(f"SELECT {', '.join(chunk)} FROM {qualified}")).fetchall()[0]  # noqa: S608 - identifiers come from information_schema
        values.extend(row)
    return values


def _measure_pairs(
    conn: Connection,
    qualified: str,
    names: Sequence[str],
    row_count: int,
    batch_size: int,
) -> dict[frozenset[str], int]:
    """
    ``count(DISTINCT (a, b))`` for every column pair - the one measurement two families
    are built on.

    A pair count answers both questions at once: ``{a,b}`` is a key exactly when the count
    equals the row count, and ``a -> b`` holds exactly when it equals ``distinct(a)``.
    """
    pairs = [frozenset(pair) for pair in combinations(names, 2)]
    if not pairs or row_count == 0:
        return {}

    aggregates = [_combination_expr(sorted(pair)) for pair in pairs]
    values = _run_aggregates(conn, qualified, aggregates, batch_size)
    return dict(zip(pairs, (int(value or 0) for value in values), strict=True))


# --------------------------------------------------------------------------------------
# Key search (decision E1) and dependency derivation
# --------------------------------------------------------------------------------------

MAX_UCC_LEVEL = 3


def find_uccs(
    conn: Connection,
    qualified: str,
    profile: TableProfile,
    max_level: int = MAX_UCC_LEVEL,
    batch_size: int = 40,
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

    Returns:
        The UCCs found, and whether the search ran out of levels without finding one.
        The second value becomes ``table_has_no_ucc_le3`` and the review flag from **E1** -
        an honest "unknown" state has to exist in the training data, or the model learns
        that a key is always available.
    """
    names = [column.name for column in profile.columns]
    rows = profile.row_count
    if rows == 0:
        return [], True

    uccs = [frozenset({name}) for name in names if profile.distinct[name] == rows]

    if max_level >= 2:
        uccs += [
            pair
            for pair, count in profile.pair_distinct.items()
            if count == rows and not any(ucc <= pair for ucc in uccs)
        ]

    if max_level >= 3:
        candidates = [
            frozenset(triple)
            for triple in combinations(names, 3)
            if _cardinality_allows(profile, triple, rows)
            and not any(ucc <= frozenset(triple) for ucc in uccs)
        ]
        if candidates:
            aggregates = [_combination_expr(sorted(triple)) for triple in candidates]
            values = _run_aggregates(conn, qualified, aggregates, batch_size)
            uccs += [
                triple
                for triple, count in zip(candidates, values, strict=True)
                if int(count or 0) == rows
            ]

    return uccs, not uccs


def _cardinality_allows(profile: TableProfile, columns: Iterable[str], rows: int) -> bool:
    """
    Necessary condition for uniqueness: the distinct counts must multiply out to at least
    the row count. Pure arithmetic on numbers already measured, and it removes most
    triples before any SQL is issued.
    """
    product = 1
    for name in columns:
        product *= max(profile.distinct[name], 1)
        if product >= rows:
            return True
    return False


def find_dependencies(profile: TableProfile) -> list[tuple[str, str]]:
    """
    Single-attribute dependencies ``a -> b`` that hold in the data.

    ``a -> b`` holds when adding ``b`` does not split any group of ``a``, i.e. when
    ``distinct(a, b) == distinct(a)``. Dependencies out of a column whose values are all
    distinct are dropped: they hold trivially and would swamp every count with noise.
    """
    rows = profile.row_count
    found = []
    for pair, both in profile.pair_distinct.items():
        left, right = sorted(pair)
        for determinant, dependent in ((left, right), (right, left)):
            if profile.distinct[determinant] == rows:
                continue  # a key determines everything - not a finding
            if both == profile.distinct[determinant]:
                found.append((determinant, dependent))
    return found


# --------------------------------------------------------------------------------------
# Feature assembly
# --------------------------------------------------------------------------------------


def build_features(
    conn: Connection,
    catalog: str,
    schema: str,
    table: str,
) -> pd.DataFrame:
    """
    One feature row per column of ``table``.

    Table-level features are repeated on every row: the model sees a single row per call
    and cannot aggregate anything itself - dropping table-level aggregates as "derivable"
    once cost 0.22 F1 (finding 1.1).
    """
    qualified = f"{catalog}.{schema}.{table}"
    profile = profile_table(conn, catalog, schema, table)
    names = [column.name for column in profile.columns]
    rows = profile.row_count

    uccs, no_key_found = find_uccs(conn, qualified, profile)
    prime = frozenset().union(*uccs) if uccs else frozenset()
    dependencies = find_dependencies(profile)

    repeating = repeating_group_columns(names)
    freetext = {name for name in names if is_freetext_name(name)}

    violating = {
        name
        for name in names
        if name not in freetext
        and profile.list_like_ratio[name] >= LIST_VALUE_MIN_SHARE
        and _mean_token_length(profile, name) <= LIST_TOKEN_MAX_MEAN_LENGTH
    }

    classes = classify_dependencies(dependencies, uccs, prime)
    determines = dict.fromkeys(names, 0)
    depends_on = dict.fromkeys(names, 0)
    for determinant, dependent in dependencies:
        determines[determinant] += 1
        depends_on[dependent] += 1

    column_count = len(names) or 1
    possible_pairs = column_count * (column_count - 1) or 1
    table_level = {
        "table_row_count": rows,
        "table_column_count": len(names),
        "table_max_list_like_ratio": max(profile.list_like_ratio.values(), default=0.0),
        "table_ratio_list_like_columns": len(violating) / column_count,
        "table_repeating_group_ratio": len(repeating) / column_count,
        "table_has_no_ucc_le3": int(no_key_found),
        "table_candidate_key_count": len(uccs),
        "table_key_size": min((len(key) for key in uccs), default=0),
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
                "col_unique_ratio": profile.distinct[name] / rows if rows else 0.0,
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
