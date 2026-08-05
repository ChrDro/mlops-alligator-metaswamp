"""
Cross-table features for the foreign-key model.

Why this module exists
----------------------
A foreign key is not a property of a column. It is a property of a *pair* of columns in
two different tables: the child's values reference the parent's key. Until now every
feature fed to the fk model was computed inside a single column or a single table - null
ratio, unique ratio, ordinal position, table row count, four name regexes. Nothing in
the feature matrix could see another table, so the model had to approximate a relational
property from local shape and naming. Out-of-fold error analysis on `fk_target` showed
what that cost:

- The third most important feature was `name_length`. Not a signal, a proxy.
- The misses were natural keys and non-standard key names: `winning_pilot`,
  `airline_iata`, `flight_ref`, plus `aid` / `cid` / `did` / `itemid` / `songid` /
  `driverid` / `region_code` / `dept_code`. The four existing `LIKE '%_id'`-style
  regexes catch none of those, but "some table in this database has a primary key with
  this exact name" catches most of them.

Measured effect on `fk_target` (5-fold StratifiedGroupKFold grouped by `database`,
default LightGBM, pooled out-of-fold):

    baseline, 30 features          F1 0.696 +/- 0.082
    + these 7 features            F1 0.805 +/- 0.057
    false negatives                404 -> 284
    false positives                703 -> 426

197 previously-missed foreign keys are now caught, and they are the predicted classes:
`itemid`, `songid`, `driverid`, `productid`, `msid` (no-underscore ids) and
`region_code`, `category_code`, `department_code`, `country_code`, `type_code`
(code-suffix keys).

Measured and rejected - do not re-add without new evidence
----------------------------------------------------------
Three ideas looked sound and did not survive the ablation:

1. *Value-range containment* (`value_range_within_candidate_parent`), plus candidate
   parent type compatibility, parent row-count ratio and parent count. A cheap stand-in
   for an inclusion dependency. Worth +0.002 F1, because only about half of
   `min_value`/`max_value` parse as numeric - the varchar columns, which are the
   majority, never fire. Not worth plumbing `min_value`/`max_value` through the serving
   path for.
2. *A parent-side suppressor* (`is_unique_and_name_repeated_elsewhere`): unique here and
   the same name appears in other tables, so probably the referenced key rather than a
   referencing one. It ranked 30th of 39 and *hurt*: removing it gained +0.007 F1 and
   cut the fold spread from 0.064 to 0.057. Role confusion is still the largest
   remaining error class - 43% of the surviving false positives are really PK, composite
   PK or composite FK columns - but this framing does not fix it.
3. *A short-abbreviation flag* (`name_is_short_id_abbrev`, for `aid` / `cid` / `dno`).
   Also harmful; `name_unique_in_other_table` already covers those columns through their
   parent table, which is the stronger signal.

Leave-one-out over the seven that remain costs between 0.007 and 0.017 F1 each, so none
of them is redundant.

Scope of the grouping key
-------------------------
`database` is the reference scope, matching the grouping key of every cross-validation
split in the training scripts. Two databases may each have a `users` table with an `id`
column and no relationship between them, so looking across databases would manufacture
parents that cannot exist.

Serving
-------
Only `database`, `table_name`, `column_name` and `is_unique` are needed, all of which
the Prefect extractor (prefect/pk_fk_pipeline.py) already has for every column of a
schema before it profiles anything. These features are not wired into the serving path
yet: that change also touches the Pydantic request schema, the model contract test, the
curl fixture and the regenerated monitoring CSVs, so it is deliberately separate from
proving the features pay off offline.
"""

import pandas as pd


# Suffixes stripped to get at the "thing" a column names: `player_id` -> `player`,
# `artistid` -> `artist`, `dept_code` -> `dept`. Longest first, so `_id` wins over `id`
# and the stem of `player_id` is not `player_`.
ID_SUFFIXES = (
    "_id",
    "_key",
    "_num",
    "_code",
    "_ref",
    "_no",
    "_fk",
    "id",
    "key",
    "num",
    "code",
    "ref",
    "no",
)

# A stem shorter than this carries no matchable information (`cid` -> `c` would match
# every table), so the full name is kept instead. Those columns are reached through
# `name_unique_in_other_table` rather than through table-name matching.
MIN_STEM_LENGTH = 2

# Substring matching below this length produces noise (`no` matches `notes`).
MIN_FUZZY_STEM_LENGTH = 3

# Non-underscored id endings, for `artistid` / `songid` / `driverid`.
BARE_ID_SUFFIXES = ("id", "no")

# Suffixes of the "not called id but is a key" kind, for `dept_code` / `emp_num` /
# `flight_ref`.
CODE_SUFFIXES = ("_no", "_num", "_code", "_key", "_ref", "no", "num", "code", "ref")

# The seven features that survived the ablation described in the module docstring.
CROSS_TABLE_FEATURES = [
    "n_other_tables_with_same_column_name",
    "name_unique_in_other_table",
    "is_non_unique_and_name_unique_elsewhere",
    "name_references_other_table_exact",
    "name_references_other_table_fuzzy",
    "name_ends_with_id_no_underscore",
    "name_ends_with_code_or_num",
]

REQUIRED_COLUMNS = (
    "database",
    "table_name",
    "column_name",
    "is_unique",
)


def column_stem(column_name: str) -> str:
    """Strip one key-ish suffix off a column name.

    `player_id` -> `player`, `artistid` -> `artist`, `dept_code` -> `dept`. Falls back to
    the untouched name when stripping would leave something too short to match anything
    (`cid` -> `c`).
    """
    name = str(column_name).lower()
    for suffix in ID_SUFFIXES:
        if name.endswith(suffix):
            stem = name.removesuffix(suffix).rstrip("_")
            return stem if len(stem) >= MIN_STEM_LENGTH else name
    return name


def table_name_variants(table_name: str) -> set[str]:
    """The table name plus its naive singular/plural forms.

    `orders`/`order` and `categories`/`category` are the same entity as far as a column
    called `order_id` or `category_id` is concerned, and this dataset uses both
    conventions.
    """
    name = str(table_name).lower()
    variants = {name}
    if name.endswith("ies"):
        variants.add(name.removesuffix("ies") + "y")
    if name.endswith("s"):
        variants.add(name.removesuffix("s"))
    else:
        variants.add(name + "s")
    return variants


class _DatabaseIndex:
    """Lookups over one database's columns, built once and queried per row."""

    def __init__(self, frame: pd.DataFrame) -> None:
        # column name (lowercased) -> tables that have a column with that name
        self.tables_by_column: dict[str, set[str]] = {}
        # column name -> tables where that column is unique and non-null, i.e. tables for
        # which it is a single-column primary key candidate
        self.unique_tables_by_column: dict[str, set[str]] = {}
        # every singular/plural variant of every table name in this database
        self.table_variants: set[str] = set()

        for row in frame.itertuples(index=False):
            table = str(row.table_name).lower()
            column = str(row.column_name).lower()

            self.table_variants |= table_name_variants(table)
            self.tables_by_column.setdefault(column, set()).add(table)
            if int(row.is_unique) == 1:
                self.unique_tables_by_column.setdefault(column, set()).add(table)

    def parent_tables(self, table: str, column: str) -> set[str]:
        """Other tables in which a same-named column is a primary key candidate.

        This is the cheap stand-in for an inclusion dependency: a real check would
        compare value sets, which needs a scan per column pair. A same-named unique
        column in a sibling table is the parent a human would guess, and it is
        computable from metadata alone.
        """
        return self.unique_tables_by_column.get(column, set()) - {table}


def _row_features(index: _DatabaseIndex, table: str, column: str, is_unique: int) -> dict:
    """The seven cross-table features for one column."""
    stem = column_stem(column)
    other_tables_with_name = index.tables_by_column.get(column, set()) - {table}
    unique_elsewhere = bool(index.parent_tables(table, column))

    # A column naming its own table says nothing about a reference, so the own table's
    # variants are excluded before matching.
    other_variants = index.table_variants - table_name_variants(table)
    references_exact = stem in other_variants
    references_fuzzy = references_exact or (
        len(stem) >= MIN_FUZZY_STEM_LENGTH and any(stem in variant for variant in other_variants)
    )

    return {
        "n_other_tables_with_same_column_name": len(other_tables_with_name),
        "name_unique_in_other_table": int(unique_elsewhere),
        # The child side of a relationship: the values repeat here but are a key
        # somewhere else. This is what "is a foreign key" actually looks like.
        "is_non_unique_and_name_unique_elsewhere": int(unique_elsewhere and is_unique == 0),
        "name_references_other_table_exact": int(references_exact),
        "name_references_other_table_fuzzy": int(references_fuzzy),
        "name_ends_with_id_no_underscore": int(
            column.endswith(BARE_ID_SUFFIXES) and not column.endswith(("_id", "_no")),
        ),
        "name_ends_with_code_or_num": int(column.endswith(CODE_SUFFIXES)),
    }


def add_cross_table_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with the cross-table feature columns appended.

    Call this on the raw training frame, before one-hot encoding, because it needs the
    identity columns. The row order and index of the input are preserved, so the result
    lines up with any target column taken from `df`.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        msg_missing_columns = f"add_cross_table_features needs columns {missing}"
        raise KeyError(msg_missing_columns)

    # groupby(sort=False) yields groups in order of first appearance and keeps row order
    # inside each group, but the collected records follow group order rather than the
    # original row order - so the source index is collected alongside them and used to
    # realign the result.
    records: list[dict] = []
    source_index: list = []
    for _, database_frame in df.groupby("database", sort=False):
        index = _DatabaseIndex(database_frame)
        source_index.extend(database_frame.index.tolist())
        records.extend(
            _row_features(
                index,
                str(row.table_name).lower(),
                str(row.column_name).lower(),
                int(row.is_unique),
            )
            for row in database_frame.itertuples(index=False)
        )

    features = pd.DataFrame(records, index=source_index).reindex(df.index)
    return pd.concat([df, features], axis=1)
