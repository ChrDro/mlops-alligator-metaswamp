"""
Synthetic schemas for the normal-form training set (Phase 2d of TASK_3_PLAN.md).

Why this exists alongside the TPC-H recipes
-------------------------------------------
2b measured the bottleneck exactly: F1 is 0.95 when tables are split by name and 0.77 when
they are split by *recipe*. The gap is memorised shape, and it closes only with more
shapes. TPC-H has eight base relations, which caps it at a few dozen - the plan's target is
**300+ source schemas**, and the number of schemas is the harder half of it.

So: TPC-H supplies realism, this module supplies coverage. Neither on its own is enough. A
model trained only on synthetic schemas learns a generator; one trained only on TPC-H
learns TPC-H.

How the dependencies are made exact
-----------------------------------
Every attribute is a deterministic function of something, and that is what makes the
declared FD set true by construction rather than by hope:

* **key columns** are a positional decomposition of the row number, so they are unique
  together and non-unique apart,
* **independent attributes** are ``hash(row_id, salt) mod cardinality`` - unrelated to
  everything else,
* **dependent attributes** are ``hash(determinant_value, salt) mod cardinality``, which
  makes ``determinant -> dependent`` hold exactly, for every row, at any scale.

The last one is the whole trick. A column that is a function of another column satisfies
the dependency by arithmetic; no sampling, no tolerance, nothing to verify by luck.

What is deliberately *not* controlled
-------------------------------------
Accidental dependencies between independent attributes. Different salts make them
unlikely, not impossible - and the validation harness (2c) already asks whether the
discovered dependencies move the label. A generator that promises what the harness is
there to check would be checking itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from nf_generator import TableSpec
from nf_labeling import FunctionalDependency, normal_form, normalise_fds


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


# --------------------------------------------------------------------------------------
# Variation axes (TASK_3_PLAN.md, 2d)
# --------------------------------------------------------------------------------------

ROW_COUNTS: tuple[int, ...] = (500, 5_000, 50_000)
COLUMN_COUNTS: tuple[int, ...] = (5, 10, 20, 40)
NULL_RATES: tuple[float, ...] = (0.0, 0.05, 0.30)
KEY_SIZES: tuple[int, ...] = (1, 2, 3)
NAMINGS: tuple[str, ...] = ("real", "convention", "obfuscated")

# Separators for the 0NF injections, cycled so no separator correlates with anything else.
SEPARATORS: tuple[str, ...] = (",", ";", "|")

# Column-name vocabularies. Three naming regimes, because the current model leans on name
# features (finding 1.2) and a third of the tables carrying meaningless names is what stops
# that from working.
_REAL_NOUNS: tuple[str, ...] = (
    "customer",
    "invoice",
    "product",
    "region",
    "channel",
    "status",
    "category",
    "vendor",
    "contract",
    "shipment",
    "account",
    "campaign",
    "branch",
    "currency",
    "employee",
    "project",
    "device",
    "tariff",
    "segment",
    "warehouse",
)
_REAL_QUALIFIERS: tuple[str, ...] = (
    "code",
    "name",
    "type",
    "group",
    "class",
    "level",
    "state",
    "label",
    "kind",
    "band",
)


def _deterministic_index(seed: int, salt: int, modulus: int) -> int:
    """A small reproducible pseudo-random index. No RNG state, so no run-to-run drift."""
    return ((seed * 2_654_435_761) ^ (salt * 40_503)) % modulus


# --------------------------------------------------------------------------------------
# SQL building blocks
# --------------------------------------------------------------------------------------


def _hashed(expression: str, salt: int, cardinality: int) -> str:
    """
    ``hash(expression, salt) mod cardinality``, always non-negative.

    Two Trino details are baked in. ``xxhash64`` returns *varbinary*, so it needs
    ``from_big_endian_64`` before it is a number at all; and the number it then gives can
    be negative, where ``abs()`` on the minimum bigint throws - hence the double modulo
    instead of an absolute value.
    """
    hashed = f"from_big_endian_64(xxhash64(to_utf8(CAST({expression} AS VARCHAR) || '{salt}')))"
    return f"mod(mod({hashed}, {cardinality}) + {cardinality}, {cardinality})"


def _nullable(expression: str, salt: int, null_rate: float, gate_on: str = "row_id") -> str:
    """
    Punch NULLs into a column at roughly ``null_rate``.

    Never applied to a key column or a determinant: a NULL in a determinant would collapse
    distinct groups into one and quietly change what the FD says.

    ``gate_on`` decides *which* rows go NULL and has to match how the column is built. A
    dependent column must be gated on its own determinant, not on ``row_id`` - otherwise
    two rows sharing a determinant value get independent draws, one keeps its value, the
    other goes NULL, and the declared FD is broken. Gated on the determinant, a whole FD
    group goes NULL together and the dependency survives.
    """
    if null_rate <= 0:
        return expression
    threshold = int(null_rate * 100)
    gate = _hashed(gate_on, salt, 100)
    return f"CASE WHEN {gate} < {threshold} THEN NULL ELSE {expression} END"


# --------------------------------------------------------------------------------------
# Schema construction
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticSchema:
    """One generated relation: its columns, what holds in it, and the SQL that builds it."""

    schema_id: int
    target: int
    attributes: tuple[str, ...]
    fds: tuple[FunctionalDependency, ...]
    sql: str
    violates_1nf: bool
    params: dict[str, object] = field(default_factory=dict)


def _column_names(count: int, naming: str, seed: int) -> list[str]:
    """
    Column names under one of three regimes.

    ``obfuscated`` avoids digits on purpose: a numeric suffix is exactly what the
    repeating-group detector looks for, so numbered names would turn every obfuscated
    table into a fake 1NF violation.
    """
    if naming == "obfuscated":
        return [f"x{_letters(index)}" for index in range(count)]
    if naming == "convention":
        return [f"col_{_letters(index)}" for index in range(count)]

    names, used = [], set()
    for index in range(count):
        noun = _REAL_NOUNS[_deterministic_index(seed, index, len(_REAL_NOUNS))]
        qualifier = _REAL_QUALIFIERS[_deterministic_index(seed, index + 7, len(_REAL_QUALIFIERS))]
        candidate = f"{noun}_{qualifier}"
        suffix = 0
        while candidate in used:
            suffix += 1
            qualifier = _REAL_QUALIFIERS[(_deterministic_index(seed, index + 7, 10) + suffix) % 10]
            candidate = f"{noun}_{qualifier}"
            if suffix > len(_REAL_QUALIFIERS):
                candidate = f"{noun}_{_letters(index)}"
                break
        used.add(candidate)
        names.append(candidate)
    return names


def _letters(index: int) -> str:
    """0 -> 'a', 25 -> 'z', 26 -> 'aa'. Digit-free, so no fake repeating groups."""
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("a") + remainder) + name
    return name


def _key_expressions(key_columns: Sequence[str], n_rows: int) -> tuple[list[str], list[int]]:
    """
    Decompose the row number into ``k`` columns that are unique together, not apart.

    Sizes are chosen so their product just covers the row count - the key is exactly a
    key, with no room left over that would make a proper subset unique too.
    """
    count = len(key_columns)
    if count == 1:
        return ["row_id"], [n_rows]

    size = max(2, round(n_rows ** (1 / count)) + 1)
    sizes = [size] * count
    while _product(sizes) < n_rows:
        sizes[0] += 1

    expressions, divisor = [], 1
    for index in range(count - 1, -1, -1):
        # Trino has no `div`; `/` on bigints is integer division.
        inner = "row_id" if divisor == 1 else f"row_id / {divisor}"
        expressions.insert(0, f"mod({inner}, {sizes[index]})")
        divisor *= sizes[index]
    return expressions, sizes


# Trino refuses to build a `sequence` longer than this, so wider row counts are made by
# crossing two shorter ones.
_MAX_SEQUENCE = 1_000


def _row_source(n_rows: int) -> str:
    """
    A ``row_id`` running from 1 to ``n_rows``.

    Not a single ``sequence(1, n)``: Trino caps that at 10.000 entries, which the 50.000-row
    tier of the plan's row-count axis walks straight into. Two crossed sequences have no
    such ceiling, and the trailing ``WHERE`` trims the last partial chunk so the count is
    exact rather than rounded up.
    """
    chunk = min(n_rows, _MAX_SEQUENCE)
    chunks = -(-n_rows // chunk)
    return (
        "FROM (\n"  # noqa: S608 - the only values interpolated below are row counts
        f"  SELECT (a.hi * {chunk}) + b.lo AS row_id\n"
        f"  FROM UNNEST(sequence(0, {chunks - 1})) AS a(hi)\n"
        f"  CROSS JOIN UNNEST(sequence(1, {chunk})) AS b(lo)\n"
        ") t\n"
        f"WHERE t.row_id <= {n_rows}"
    )


def _product(values: Iterable[int]) -> int:
    total = 1
    for value in values:
        total *= value
    return total


# How many dependencies of one kind a schema may carry. One each was a label fingerprint:
# with exactly one transitive dependency for 2NF and exactly one of each for 1NF, the plain
# count of discovered dependencies read the class off almost directly - `table_fd_count`
# alone reached 0.6655 weighted F1 against a 0.6 limit, and the signature check in 2c
# failed on it. Drawing 1 to 3 of each makes the ranges overlap (2NF 1-3, 1NF 2-6), so no
# single threshold separates them any more, while the structural difference - *where* the
# determinant sits - is untouched.
#
# Worth knowing why the bug in the NULL gate hid this: while 130 tables declared FDs that
# did not hold in the data, their discovered counts were noise, and the fingerprint only
# appeared once the labels were correct. The measured F1 rose from 0.4011 to 0.6655 because
# the data got *better*.
MAX_COUNTED_DEPENDENCIES = 6


def _dependency_counts(structure: int, schema_id: int, available: int) -> tuple[int, int]:
    """
    How many transitive and partial dependencies this schema gets, and it varies on purpose.

    A *total* is drawn first and then split, rather than drawing each kind on its own. That
    is what makes the ranges line up: a 2NF schema carries 1 to 6 dependencies and a 1NF one
    2 to 6, so the plain count no longer separates them. Drawn per kind, 2NF would top out
    where 1NF starts and a single threshold would still find the seam.

    A 1NF schema always keeps at least one of each - the partial dependency is what makes it
    1NF, and the transitive one is inherited from the 2NF structure it is built on.

    Returns:
        ``(n_transitive, n_partial)``; both zero for a 3NF structure, which has neither.
    """
    if structure >= 3:
        return 0, 0

    minimum = 1 if structure == 2 else 2
    spread = MAX_COUNTED_DEPENDENCIES - minimum + 1
    total = minimum + _deterministic_index(schema_id, 71, spread)
    # One column carries the determinant, every dependency needs one of its own.
    total = min(total, max(minimum, available - 1))

    if structure == 2:
        return total, 0
    n_partial = 1 + _deterministic_index(schema_id, 73, total - 1)
    return total - n_partial, n_partial


def build_schema(
    schema_id: int,
    target: int,
    n_attributes: int,
    key_size: int,
    n_rows: int,
    null_rate: float,
    naming: str,
    base: int | None = None,
) -> SyntheticSchema:
    """
    Build one relation whose highest normal form is exactly ``target``.

    The construction per class, each adding to the one above it:

    * **3NF** - the key determines every other attribute, and nothing else holds.
    * **2NF** - plus a transitive chain: one non-key attribute determines another. Its
      determinant is no superkey and its dependent is not prime, which is precisely a 3NF
      violation and nothing more.
    * **1NF** - plus a partial dependency: *part* of a composite key determines a non-key
      attribute. Needs ``key_size >= 2``, which is why 1NF is only ever requested with one.
    * **0NF** - a list-valued column on top of *any* of the above. Atomicity is a property
      of values, so this is the one part no FD set can express; it is passed to the label
      function separately.

    Args:
        base: Which FD structure to build under a 0NF label - 1, 2 or 3. Only meaningful
            when ``target`` is 0, and it matters: if every 0NF table also carried a partial
            dependency, "0NF" and "has a partial dependency" would move together and the
            model could read one off the other. The same reasoning put the TPC-H 0NF
            injections on 3NF *and* 2NF parents.

    The resulting label is verified against ``nf_labeling`` before the schema is returned -
    a constructor that believes it built a 1NF table but wired up a superkey would
    otherwise ship a wrong label silently.

    Raises:
        ValueError: The construction cannot be built as asked, or did not produce the
            requested normal form.
    """
    structure = base if target == 0 else target
    if structure <= 1 and key_size < 2:
        message = f"a partial dependency needs a composite key, got key_size={key_size}"
        raise ValueError(message)

    # The 0NF list column replaces an ordinary attribute instead of joining it, so that
    # `table_column_count` stays uninformative. Added on top it would make the count
    # itself a perfect label hint - 6/11/21/41 against 5/10/20/40 - which is precisely the
    # kind of generator signature Phase 2 exists to remove.
    names = _column_names(n_attributes - (1 if target == 0 else 0), naming, schema_id)
    key_columns = names[:key_size]
    others = names[key_size:]
    needed = {3: 1, 2: 2, 1: 3}[structure]
    if len(others) < needed:
        message = (
            f"{n_attributes} attributes with a {key_size}-column key leave {len(others)} "
            f"non-key attributes; a {structure}NF structure needs {needed}"
        )
        raise ValueError(message)

    key_expressions, key_sizes = _key_expressions(key_columns, n_rows)
    select: dict[str, str] = dict(zip(key_columns, key_expressions, strict=True))
    fds: list[tuple[frozenset[str], frozenset[str]]] = [
        (frozenset(key_columns), frozenset(others)),
    ]

    # Attributes hanging off the key: independent of each other, each a fresh hash.
    for position, name in enumerate(others):
        cardinality = 2 + _deterministic_index(schema_id, position, min(64, max(4, n_rows // 8)))
        expression = _hashed("row_id", schema_id * 131 + position, cardinality)
        select[name] = _nullable(expression, schema_id * 977 + position, null_rate)

    n_transitive, n_partial = _dependency_counts(structure, schema_id, len(others))

    # 2NF and below: a transitive chain, key -> determinant -> dependents.
    if structure <= 2:
        determinant = others[0]
        determinant_cardinality = max(4, min(n_rows // 4, 64))
        select[determinant] = _hashed("row_id", schema_id * 31 + 3, determinant_cardinality)
        for offset, dependent in enumerate(others[1 : 1 + n_transitive]):
            # The determinant's expression, not its alias: Trino cannot reference an output
            # alias from elsewhere in the same SELECT list. Inlining it also keeps the
            # dependency exact - the dependent is a function of the very same value.
            select[dependent] = _nullable(
                _hashed(f"({select[determinant]})", schema_id * 37 + 5 + offset * 3, 6 + offset),
                schema_id * 41 + offset,
                null_rate,
                gate_on=f"({select[determinant]})",
            )
            fds.append((frozenset({determinant}), frozenset({dependent})))

    # 1NF and below: half the key determines non-key attributes.
    if structure <= 1:
        partial_source = key_columns[0]
        first = 1 + n_transitive
        for offset, partial_target in enumerate(others[first : first + n_partial]):
            select[partial_target] = _nullable(
                _hashed(
                    f"({select[partial_source]})", schema_id * 53 + 7 + offset * 3, 10 + offset
                ),
                schema_id * 59 + offset,
                null_rate,
                gate_on=f"({select[partial_source]})",
            )
            fds.append((frozenset({partial_source}), frozenset({partial_target})))

    # Decoy columns, on the attributes no dependency structure claimed. Both are trivial
    # dependencies dressed up as real ones, and both cost a normal form on real TPC-H before
    # `find_dependencies` learned to drop them. They are here so that a regression in those
    # guards fails the label-versus-discovered check on generated data instead of on someone's
    # database - the training set previously contained neither phenomenon, because the recipes
    # had removed the columns that produce them.
    #
    # Neither changes the label, and neither needs a declared FD beyond `key -> others`: a
    # constant column is determined by the key like any other attribute, and a near-unique
    # column is simply a non-key attribute with a lot of values.
    spare = others[1 + n_transitive + n_partial :]
    constant_column, near_unique_column = None, None
    if len(spare) >= 1 and _deterministic_index(schema_id, 83, 3) == 0:
        constant_column = spare[0]
        select[constant_column] = "0"
    if len(spare) >= 2 and _deterministic_index(schema_id, 89, 3) == 0:
        near_unique_column = spare[1]
        # One collision per ~20 rows: high enough to look like a key, not unique.
        select[near_unique_column] = f"mod(row_id, {max(2, n_rows - n_rows // 20)})"

    # 0NF: a list-valued column. Never nulled - the detector measures a share of non-null
    # values, and thinning them out would only blur what it is being tested on.
    violates_1nf = target == 0
    if violates_1nf:
        separator = SEPARATORS[schema_id % len(SEPARATORS)]
        list_column = f"{others[-1]}_list" if naming != "obfuscated" else f"x{_letters(len(names))}"
        pieces = [
            f"CAST({_hashed('row_id', schema_id * 71 + offset, 40)} AS VARCHAR)"
            for offset in range(3)
        ]
        select[list_column] = " || ".join(
            piece if index == 0 else f"'{separator}' || {piece}"
            for index, piece in enumerate(pieces)
        )
        names = [*names, list_column]
        fds.append((frozenset(key_columns), frozenset({list_column})))

    normalised = normalise_fds(fds, names)
    actual = normal_form(names, normalised, violates_1nf)
    if actual != target:
        message = f"schema {schema_id} was built for {target}NF but computes as {actual}NF"
        raise ValueError(message)

    projection = ",\n       ".join(f'{select[name]} AS "{name}"' for name in names)
    sql = f"SELECT {projection}\n{_row_source(n_rows)}"

    return SyntheticSchema(
        schema_id=schema_id,
        target=target,
        attributes=tuple(names),
        fds=tuple(normalised),
        sql=sql,
        violates_1nf=violates_1nf,
        params={
            "source": "synthetic",
            "n_attributes": len(names),
            "key_size": key_size,
            "key_cardinalities": key_sizes,
            "row_count": n_rows,
            "n_transitive": n_transitive,
            "n_partial": n_partial,
            "constant_column": constant_column,
            "near_unique_column": near_unique_column,
            "null_rate": null_rate,
            "naming": naming,
            "base": structure,
        },
    )


# --------------------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------------------

# How many schemas to build. The plan's target is 300+ *source schemas*, and each of these
# is one - unlike the TPC-H recipes, which all descend from the same eight relations.
SCHEMA_COUNT = 300


def _fit(structure: int, available: int, desired_key: int) -> tuple[int, int]:
    """
    Fit a dependency structure into the columns available, degrading it if it does not.

    A partial dependency needs a composite key *and* three non-key attributes, so it does
    not fit into five columns once one of them is a list. Degrading the structure is safe
    only where the label does not depend on it - which is exactly the 0NF case, where the
    label comes from atomicity and the FD structure underneath is free. For 1NF, 2NF and
    3NF the caller has already guaranteed the width.

    Returns:
        The structure actually built and the key size to build it with.
    """
    for candidate in (structure, 2, 3):
        minimum_key = 2 if candidate <= 1 else 1
        needed = {3: 1, 2: 2, 1: 3}[candidate]
        if available - minimum_key >= needed:
            key_size = max(minimum_key, min(desired_key, available - needed))
            return candidate, key_size
    message = f"{available} columns are too few for any dependency structure"
    raise ValueError(message)


def build_synthetic_specs(count: int = SCHEMA_COUNT) -> list[TableSpec]:
    """
    Sweep the variation axes deterministically, one schema per step.

    Deterministic rather than random: regenerating has to produce the same data, or the
    label verification in 2c compares against a table that no longer exists. The axes are
    walked at different strides so they decorrelate - if row count and normal form moved
    together, row count would become a label hint of exactly the kind Phase 2 exists to
    remove.
    """
    specs: list[TableSpec] = []
    for schema_id in range(count):
        target = schema_id % 4
        # A 0NF table gets an FD structure of its own, cycling through 1/2/3 - otherwise
        # "not atomic" and "has a partial dependency" would always travel together.
        base = 1 + (schema_id // 4) % 3 if target == 0 else target
        structure = base if target == 0 else target

        n_attributes = COLUMN_COUNTS[(schema_id // 7) % len(COLUMN_COUNTS)]
        n_rows = ROW_COUNTS[(schema_id // 11) % len(ROW_COUNTS)]
        null_rate = NULL_RATES[(schema_id // 13) % len(NULL_RATES)]
        naming = NAMINGS[(schema_id // 3) % len(NAMINGS)]

        # A 0NF table spends one column on the list, so it has one fewer to distribute.
        available = n_attributes - (1 if target == 0 else 0)
        structure, key_size = _fit(structure, available, desired_key=1 + (schema_id // 5) % 3)
        base = structure if target == 0 else base

        schema = build_schema(
            schema_id=schema_id,
            target=target,
            n_attributes=n_attributes,
            key_size=key_size,
            n_rows=n_rows,
            null_rate=null_rate,
            naming=naming,
            base=base,
        )
        specs.append(
            TableSpec(
                table_name=f"syn_{schema_id:04d}",
                source_sql=schema.sql,
                attributes=schema.attributes,
                declared_fds=schema.fds,
                violates_1nf=schema.violates_1nf,
                generation_params={
                    "recipe_id": f"synthetic_{schema_id:04d}",
                    "family": f"synthetic_{target}nf_k{key_size}_b{base}",
                    **schema.params,
                },
            ),
        )
    return specs
