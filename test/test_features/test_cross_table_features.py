"""
Tests for src/cross_table_features.py.

These features exist to fix specific, observed model errors, so the tests are written
around those cases rather than around the implementation: an abbreviated key like `stuid`
must be reachable through its parent table, a column naming its own table must not count
as a reference, and nothing may leak across database boundaries.

Two structural properties matter as much as the feature values, because breaking either
would silently corrupt training rather than fail loudly:

- row alignment: the returned frame must line up with the input's index, or every feature
  ends up attached to the wrong column
- database isolation: a parent may never be found in a different database
"""

import sys
from pathlib import Path

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cross_table_features import (
    CROSS_TABLE_FEATURES,
    SYNTHETIC_SCOPE_COLUMN,
    add_cross_table_features,
    column_stem,
    scope_augmented_variants,
    table_name_variants,
)


def make_row(
    database: str,
    table_name: str,
    column_name: str,
    *,
    is_unique: int = 0,
) -> dict:
    """One column-profile row, in the shape the training CSV uses.

    Only the four columns the module actually requires: the value-based features that
    needed `column_type`, `table_row_count` and `min_value`/`max_value` were measured and
    dropped, see the module docstring.
    """
    return {
        "database": database,
        "table_name": table_name,
        "column_name": column_name,
        "is_unique": is_unique,
    }


@pytest.fixture
def two_table_schema() -> pd.DataFrame:
    """A `student` parent keyed by `stuid`, and a `has_pet` child referencing it.

    Modelled on the real rows the fk model was missing: the child column has the same
    abbreviated name as the parent's key and matches no `%_id` pattern.
    """
    return pd.DataFrame(
        [
            make_row("uni", "student", "stuid", is_unique=1),
            make_row("uni", "student", "lname"),
            make_row("uni", "has_pet", "stuid"),
            make_row("uni", "has_pet", "petid"),
        ],
    )


class TestColumnStem:
    @pytest.mark.parametrize(
        ("column", "expected"),
        [
            ("player_id", "player"),
            ("artistid", "artist"),
            ("dept_code", "dept"),
            ("emp_num", "emp"),
            ("flight_ref", "flight"),
            ("country", "country"),
        ],
    )
    def test_strips_one_key_suffix(self, column, expected):
        assert column_stem(column) == expected

    def test_keeps_the_name_when_the_stem_would_be_too_short(self):
        """`cid` -> `c` would match every table, so the full name is kept instead."""
        assert column_stem("cid") == "cid"

    def test_prefers_the_underscored_suffix(self):
        """`_id` must win over `id`, or the stem keeps a trailing underscore."""
        assert column_stem("team_id") == "team"


class TestTableNameVariants:
    @pytest.mark.parametrize(
        ("table", "expected_member"),
        [
            ("orders", "order"),
            ("order", "orders"),
            ("categories", "category"),
        ],
    )
    def test_covers_singular_and_plural(self, table, expected_member):
        assert expected_member in table_name_variants(table)


class TestChildAndParentSides:
    def test_child_column_is_marked_as_referencing(self, two_table_schema):
        """`has_pet.stuid` repeats values here but is a key in `student`: an FK."""
        result = add_cross_table_features(two_table_schema)
        child = result[(result.table_name == "has_pet") & (result.column_name == "stuid")].iloc[0]

        assert child["name_unique_in_other_table"] == 1
        assert child["is_non_unique_and_name_unique_elsewhere"] == 1

    def test_the_parent_key_is_not_marked_as_referencing(self, two_table_schema):
        """`student.stuid` is the referenced key, so the child-side flag must stay off.

        Role confusion - calling a primary key a foreign key - is the largest remaining
        error class, so this asymmetry is the one thing these features must not get wrong.
        """
        result = add_cross_table_features(two_table_schema)
        parent = result[(result.table_name == "student") & (result.column_name == "stuid")].iloc[0]

        assert parent["is_non_unique_and_name_unique_elsewhere"] == 0

    def test_abbreviated_key_is_reachable_without_any_id_pattern(self, two_table_schema):
        """`stuid` has no underscore, so only the non-underscored rule can catch it."""
        result = add_cross_table_features(two_table_schema)
        child = result[(result.table_name == "has_pet") & (result.column_name == "stuid")].iloc[0]

        assert child["name_ends_with_id_no_underscore"] == 1

    def test_an_underscored_id_is_not_counted_as_non_underscored(self):
        """`name_ends_with_id` already covers `team_id`; this feature adds the rest."""
        frame = pd.DataFrame([make_row("shop", "orders", "team_id")])
        result = add_cross_table_features(frame)

        assert result.iloc[0]["name_ends_with_id_no_underscore"] == 0

    @pytest.mark.parametrize("column", ["dept_code", "emp_num", "flight_ref"])
    def test_code_style_keys_are_flagged(self, column):
        """The `region_code` / `dept_code` family was 1 in 3 of the newly-caught FKs."""
        frame = pd.DataFrame([make_row("hr", "employee", column)])
        result = add_cross_table_features(frame)

        assert result.iloc[0]["name_ends_with_code_or_num"] == 1

    def test_a_column_with_no_counterpart_gets_zeros(self, two_table_schema):
        result = add_cross_table_features(two_table_schema)
        lonely = result[result.column_name == "lname"].iloc[0]

        assert lonely["n_other_tables_with_same_column_name"] == 0
        assert lonely["name_unique_in_other_table"] == 0
        assert lonely["is_non_unique_and_name_unique_elsewhere"] == 0


class TestTableNameReferences:
    def test_column_naming_another_table_is_flagged(self):
        frame = pd.DataFrame(
            [
                make_row("shop", "orders", "order_id", is_unique=1),
                make_row("shop", "order_items", "order_id"),
            ],
        )
        result = add_cross_table_features(frame)
        child = result[result.table_name == "order_items"].iloc[0]

        assert child["name_references_other_table_exact"] == 1

    def test_a_column_naming_only_its_own_table_is_not_flagged(self):
        """`orders.order_id` names its own table, which says nothing about a reference."""
        frame = pd.DataFrame(
            [
                make_row("shop", "orders", "order_id", is_unique=1),
                make_row("shop", "orders", "total"),
            ],
        )
        result = add_cross_table_features(frame)
        own = result[result.column_name == "order_id"].iloc[0]

        assert own["name_references_other_table_exact"] == 0

    def test_plural_parent_table_still_matches(self):
        """`customer_id` must find a `customers` table, and vice versa."""
        frame = pd.DataFrame(
            [
                make_row("shop", "customers", "id", is_unique=1),
                make_row("shop", "invoice", "customer_id"),
            ],
        )
        result = add_cross_table_features(frame)
        child = result[result.table_name == "invoice"].iloc[0]

        assert child["name_references_other_table_exact"] == 1

    def test_fuzzy_matching_ignores_stems_that_are_too_short(self):
        """A 2-character stem would match half the schema, so it must not fire."""
        frame = pd.DataFrame(
            [
                make_row("app", "notes", "body"),
                make_row("app", "other", "no"),
            ],
        )
        result = add_cross_table_features(frame)
        short = result[result.column_name == "no"].iloc[0]

        assert short["name_references_other_table_fuzzy"] == 0


class TestDatabaseIsolation:
    def test_a_parent_in_another_database_is_not_used(self):
        """Two unrelated databases both have an `id` column; neither may parent the other."""
        frame = pd.DataFrame(
            [
                make_row("db_a", "users", "id", is_unique=1),
                make_row("db_b", "orders", "id"),
            ],
        )
        result = add_cross_table_features(frame)
        other_db = result[result.database == "db_b"].iloc[0]

        assert other_db["name_unique_in_other_table"] == 0
        assert other_db["n_other_tables_with_same_column_name"] == 0

    def test_table_names_do_not_cross_databases(self):
        frame = pd.DataFrame(
            [
                make_row("db_a", "customers", "id", is_unique=1),
                make_row("db_b", "invoice", "customer_id"),
            ],
        )
        result = add_cross_table_features(frame)
        other_db = result[result.database == "db_b"].iloc[0]

        assert other_db["name_references_other_table_exact"] == 0


class TestFrameContract:
    def test_every_declared_feature_is_produced(self, two_table_schema):
        result = add_cross_table_features(two_table_schema)

        assert set(CROSS_TABLE_FEATURES) <= set(result.columns)

    def test_original_columns_and_row_count_survive(self, two_table_schema):
        result = add_cross_table_features(two_table_schema)

        assert set(two_table_schema.columns) <= set(result.columns)
        assert len(result) == len(two_table_schema)

    def test_rows_stay_aligned_when_databases_are_interleaved(self):
        """The implementation groups by database, so it must restore the input order.

        Losing this would attach every feature to the wrong column while still producing
        a plausible-looking frame - the kind of bug that surfaces only as a mysteriously
        worse F1.
        """
        frame = pd.DataFrame(
            [
                make_row("db_a", "users", "id", is_unique=1),
                make_row("db_b", "teams", "team_id", is_unique=1),
                make_row("db_a", "orders", "id"),
                make_row("db_b", "players", "team_id"),
            ],
        )
        result = add_cross_table_features(frame)

        assert result["database"].tolist() == ["db_a", "db_b", "db_a", "db_b"]
        # rows 2 and 3 are the child sides; rows 0 and 1 are the parents
        assert result["is_non_unique_and_name_unique_elsewhere"].tolist() == [0, 0, 1, 1]

    def test_a_non_default_index_is_preserved(self, two_table_schema):
        reindexed = two_table_schema.set_index(pd.Index([10, 20, 30, 40], name="row_id"))
        result = add_cross_table_features(reindexed)

        assert result.index.tolist() == [10, 20, 30, 40]

    def test_missing_input_columns_fail_loudly(self):
        with pytest.raises(KeyError, match="add_cross_table_features needs columns"):
            add_cross_table_features(pd.DataFrame({"database": ["a"]}))


class TestReferenceScope:
    """The scope decides what every feature above means, so widening it is not cosmetic."""

    @staticmethod
    def two_databases() -> pd.DataFrame:
        """The same `users.id` / `orders.id` pair in two unrelated databases."""
        return pd.DataFrame(
            [
                make_row("db_a", "users", "id", is_unique=1),
                make_row("db_a", "orders", "id"),
                make_row("db_b", "people", "id", is_unique=1),
                make_row("db_b", "invoices", "id"),
            ],
        )

    def test_a_wider_scope_finds_more_neighbours(self):
        """Grouping by database sees 1 sibling table; pooling both sees 3.

        This is the train/serve skew the augmentation exists for: serving groups by Trino
        schema, which may hold many logical databases.
        """
        frame = self.two_databases()

        narrow = add_cross_table_features(frame, scope_columns=("database",))
        frame_pooled = frame.assign(scope="all")
        wide = add_cross_table_features(frame_pooled, scope_columns=("scope",))

        assert narrow["n_other_tables_with_same_column_name"].tolist() == [1, 1, 1, 1]
        assert wide["n_other_tables_with_same_column_name"].tolist() == [3, 3, 3, 3]

    def test_a_missing_scope_column_fails_loudly(self):
        with pytest.raises(KeyError, match="add_cross_table_features needs columns"):
            add_cross_table_features(self.two_databases(), scope_columns=("subject_area",))


class TestScopeAugmentedVariants:
    def test_one_frame_per_requested_width(self):
        frame = TestReferenceScope.two_databases()
        variants = scope_augmented_variants(frame, widths=(1, None))

        assert len(variants) == 2
        for variant in variants:
            assert len(variant) == len(frame)
            assert set(CROSS_TABLE_FEATURES) <= set(variant.columns)

    def test_the_widths_really_differ(self):
        """If every variant computed the same values the augmentation would be a no-op."""
        frame = TestReferenceScope.two_databases()
        per_database, pooled = scope_augmented_variants(frame, widths=(1, None))

        assert per_database["n_other_tables_with_same_column_name"].tolist() == [1, 1, 1, 1]
        assert pooled["n_other_tables_with_same_column_name"].tolist() == [3, 3, 3, 3]

    def test_the_synthetic_scope_column_never_leaks_into_the_output(self):
        """It would otherwise reach the feature matrix as a string column and break the fit."""
        frame = TestReferenceScope.two_databases()

        for variant in scope_augmented_variants(frame, widths=(1, None)):
            assert SYNTHETIC_SCOPE_COLUMN not in variant.columns
            assert list(frame.columns) == list(variant.columns)[: len(frame.columns)]

    def test_every_variant_stays_row_aligned_with_the_input(self):
        """The training script indexes all variants with one set of fold indices."""
        frame = TestReferenceScope.two_databases()

        for variant in scope_augmented_variants(frame, widths=(1, 5, None)):
            assert variant.index.tolist() == frame.index.tolist()
            assert variant["database"].tolist() == frame["database"].tolist()
            assert variant["column_name"].tolist() == frame["column_name"].tolist()

    def test_a_width_never_splits_one_database_across_two_scopes(self):
        """Bucketing on the database code keeps each database whole, so no real database
        loses sight of its own tables at any width."""
        frame = pd.DataFrame(
            [make_row(f"db_{i}", "users", "id", is_unique=(i == 0)) for i in range(4)]
            + [make_row("db_0", "orders", "id")],
        )
        # width 2 merges db_0+db_1 and db_2+db_3; db_0's own two tables must stay together.
        (variant,) = scope_augmented_variants(frame, widths=(2,))
        child = variant[(variant.database == "db_0") & (variant.table_name == "orders")].iloc[0]

        assert child["name_unique_in_other_table"] == 1
