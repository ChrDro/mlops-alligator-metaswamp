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

The reference scope, and why it has to be augmented
---------------------------------------------------
Every feature here is relative to a *reference scope*: the set of other tables it is
allowed to look at. In training that scope is one `database` - two databases may each
have a `users` table with an `id` column and no relationship between them, so looking
across databases would manufacture parents that cannot exist.

Serving has no such boundary. `prefect/pk_fk_pipeline.py` profiles a Trino *schema*, and
`database` there is the catalog name (constant `iceberg`), so the scope is whatever the
loaders happened to put in `new_predict_data` - a flat landing area for unrelated tables.
That is a train/serve skew, and it is not a small one. Training on single-database scopes
and then serving on progressively larger ones (measured, 5-fold grouped CV on
`fk_target`):

    reference scope            tables/scope     F1
    one database (training)             7.1    0.810
    5 databases merged                 35.5    0.797
    20 databases merged               142.2    0.764
    everything in one schema         1706.0    0.666   <- worse than the 0.696 baseline

The counts inflate as the scope grows (`n_other_tables_with_same_column_name` goes from
2.14 to 14.32 on average, `name_unique_in_other_table` from 0.24 to 0.47), so a model
trained on tight scopes reads a wide scope as "everything is related".

The fix is scope augmentation, not an operational rule about how loaders must lay out
schemas: train on several scope widths at once via `scope_augmented_variants`, and the
model becomes scope-invariant.

    trained on          serve@1db  serve@5  serve@20  serve@all
    one database            0.810    0.797     0.764      0.666
    1 + 5 + pooled          0.803    0.796     0.787      0.783

That costs 0.007 F1 at the ideal layout, recovers 0.117 at the worst, and never drops
below the 0.696 no-cross-table-features baseline at any scope width. Normalising the
count by scope size instead was also measured and did not help (0.635 at serve@all).

Serving
-------
Only the scope columns plus `table_name`, `column_name` and `is_unique` are needed. The
Prefect extractor has all of them for every column of a schema before it profiles
anything, which matters in streaming mode: a run scoped to a handful of changed tables
must still compute these features against the whole schema, or a column whose parent
table was not in the run would score 0 on every feature here.
"""

from collections.abc import Sequence

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

# Columns that define the reference scope. Training groups by the logical database;
# serving passes ("database", "schema") because there `database` is the Trino catalog and
# the schema is what actually bounds a set of tables.
DEFAULT_SCOPE_COLUMNS = ("database",)

REQUIRED_COLUMNS = (
    "table_name",
    "column_name",
    "is_unique",
)

# Scope widths trained on together, in databases merged per synthetic scope. `None` pools
# every database into one scope, which is what a flat landing schema looks like. See the
# augmentation table in the module docstring for the measurements behind this choice.
SCOPE_AUGMENTATION_WIDTHS: tuple[int | None, ...] = (1, 5, None)

# Name of the synthetic scope column `scope_augmented_variants` groups by. It never
# reaches the feature matrix - the training script drops it with the other identity
# columns.
SYNTHETIC_SCOPE_COLUMN = "cross_table_scope"


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


class _ScopeIndex:
    """Lookups over one reference scope's columns, built once and queried per row."""

    def __init__(self, frame: pd.DataFrame) -> None:
        # column name (lowercased) -> tables that have a column with that name
        self.tables_by_column: dict[str, set[str]] = {}
        # column name -> tables where that column is unique and non-null, i.e. tables for
        # which it is a single-column primary key candidate
        self.unique_tables_by_column: dict[str, set[str]] = {}
        # every singular/plural variant of every table name in this scope
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


def _row_features(index: _ScopeIndex, table: str, column: str, is_unique: int) -> dict:
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


def add_cross_table_features(
    df: pd.DataFrame,
    scope_columns: Sequence[str] = DEFAULT_SCOPE_COLUMNS,
) -> pd.DataFrame:
    """Return `df` with the cross-table feature columns appended.

    Call this on the raw frame, before one-hot encoding, because it needs the identity
    columns. The row order and index of the input are preserved, so the result lines up
    with any target column taken from `df`.

    Args:
        df: Column profiles. Needs `REQUIRED_COLUMNS` plus every entry of
            `scope_columns`.
        scope_columns: Columns whose combination bounds the reference scope. Training uses
            `("database",)`; serving uses `("database", "schema")` because `database` is
            the Trino catalog there. Widening this silently changes what every feature
            means - see the scope discussion in the module docstring.
    """
    required = [*REQUIRED_COLUMNS, *scope_columns]
    missing = [column for column in required if column not in df.columns]
    if missing:
        msg_missing_columns = f"add_cross_table_features needs columns {missing}"
        raise KeyError(msg_missing_columns)

    # groupby(sort=False) yields groups in order of first appearance and keeps row order
    # inside each group, but the collected records follow group order rather than the
    # original row order - so the source index is collected alongside them and used to
    # realign the result.
    records: list[dict] = []
    source_index: list = []
    for _, scope_frame in df.groupby(list(scope_columns), sort=False):
        index = _ScopeIndex(scope_frame)
        source_index.extend(scope_frame.index.tolist())
        records.extend(
            _row_features(
                index,
                str(row.table_name).lower(),
                str(row.column_name).lower(),
                int(row.is_unique),
            )
            for row in scope_frame.itertuples(index=False)
        )

    features = pd.DataFrame(records, index=source_index).reindex(df.index)
    return pd.concat([df, features], axis=1)


def scope_augmented_variants(
    df: pd.DataFrame,
    widths: Sequence[int | None] = SCOPE_AUGMENTATION_WIDTHS,
    database_column: str = "database",
) -> list[pd.DataFrame]:
    """Featurise `df` once per reference-scope width, for scope-invariant training.

    Serving cannot promise the tight one-database-per-schema scope the training CSV has,
    so the model is shown several widths of the same rows and learns not to depend on any
    one of them. Without this, a model trained only on single-database scopes scores 0.666
    when served on a schema holding every table - below the 0.696 it gets with no
    cross-table features at all. See the module docstring for the full table.

    Args:
        df: Raw column profiles.
        widths: Databases merged into one synthetic scope per variant. `None` pools every
            database into a single scope, which is what a flat landing schema looks like.
        database_column: The real logical-database column that gets bucketed.

    Returns:
        One featurised frame per width, each keeping `df`'s original columns, row order
        and index, so they can be concatenated into a training frame or indexed with the
        same fold indices.
    """
    codes = pd.Categorical(df[database_column]).codes
    variants: list[pd.DataFrame] = []

    for width in widths:
        scoped = df.copy()
        # Bucketing by code keeps every database whole inside exactly one synthetic scope,
        # so a variant never splits a real database across two scopes.
        scoped[SYNTHETIC_SCOPE_COLUMN] = (
            "pooled" if width is None else "scope_" + (codes // width).astype(str)
        )
        featurised = add_cross_table_features(scoped, scope_columns=(SYNTHETIC_SCOPE_COLUMN,))
        variants.append(featurised.drop(columns=[SYNTHETIC_SCOPE_COLUMN]))

    return variants
