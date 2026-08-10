"""
Tests for the serving-side cross-table features in prefect/pk_fk_pipeline.py.

`add_serving_cross_table_features` is the join between the fk model's training-time
features and what the pipeline can see at prediction time, and every failure mode here is
silent: the frame still has the right shape, the insert still succeeds, and only the
predictions get worse.

The two that matter most:

- **Row alignment.** The function widens the frame with reference rows for tables this run
  did not profile, featurises the lot, then cuts back to the profiled rows. If the cut or
  the concatenation order is wrong, every feature lands on the wrong column.
- **Streaming completeness.** A run scoped to one changed table must still find parents in
  the rest of the schema. Without the reference rows those columns score 0 on every
  cross-table feature, which reads to the model as "this schema has no relationships" -
  precisely the input the model was never trained on.
"""

import pandas as pd
import pytest
from pk_fk_pipeline import CROSS_TABLE_FEATURES, add_serving_cross_table_features


class FakeUniquenessEngine:
    """Engine stand-in whose one query returns canned queue rows.

    `_known_uniqueness` reads the queue with pandas.read_sql, so the fake only has to
    survive `with engine.connect()`. Raising instead is also a case worth covering: a first
    ever run has no queue table.
    """

    def __init__(self, rows: list[dict] | None = None, fail: bool = False) -> None:
        self.rows = rows or []
        self.fail = fail
        self.queries = 0

    def connect(self):
        engine = self

        class _Ctx:
            def __enter__(self):  # noqa: ANN204
                if engine.fail:
                    message = "Table 'iceberg.predictions.queue' does not exist"
                    raise RuntimeError(message)
                engine.queries += 1
                return self

            def __exit__(self, *_exc) -> bool:
                return False

            def exec_driver_sql(self, *_args: object, **_kwargs: object):
                return None

        return _Ctx()


@pytest.fixture(autouse=True)
def _stub_read_sql(monkeypatch, request):
    """Route pandas.read_sql to the fake engine's canned rows.

    The queue lookup is the only database access in the function under test; stubbing it at
    the pandas boundary keeps the tests free of SQLAlchemy plumbing while still exercising
    the merge and fill-na logic that consumes the result.
    """
    rows = getattr(request, "param", None)

    def fake_read_sql(_query, _connection, **_kwargs: object):
        return pd.DataFrame(
            rows if rows is not None else [],
            columns=["database", "schema", "table_name", "column_name", "is_unique", "created_at"],
        )

    monkeypatch.setattr("pk_fk_pipeline.pd.read_sql", fake_read_sql)


def profiled_row(table: str, column: str, *, is_unique: int = 0, ordinal: int = 1) -> dict:
    """A row shaped like one the extraction query produces, with the fields used here."""
    return {
        "database": "iceberg",
        "schema": "new_predict_data",
        "table_name": table,
        "column_name": column,
        "column_type": "bigint",
        "is_unique": is_unique,
        "ordinal_position": ordinal,
        "table_row_count": 100,
    }


def discovered(*pairs: tuple[str, str]) -> list[tuple[str, str, str, str]]:
    """information_schema-shaped (database, schema, table, column) tuples."""
    return [("iceberg", "new_predict_data", table, column) for table, column in pairs]


class TestFullRunScope:
    def test_child_column_finds_its_parent_in_the_same_run(self):
        df_final = pd.DataFrame(
            [
                profiled_row("student", "stuid", is_unique=1),
                profiled_row("has_pet", "stuid"),
            ],
        )
        result = add_serving_cross_table_features(
            df_final,
            discovered(("student", "stuid"), ("has_pet", "stuid")),
            FakeUniquenessEngine(),
        )
        child = result[result.table_name == "has_pet"].iloc[0]

        assert child["name_unique_in_other_table"] == 1
        assert child["is_non_unique_and_name_unique_elsewhere"] == 1

    def test_the_parent_is_not_marked_as_referencing(self):
        df_final = pd.DataFrame(
            [
                profiled_row("student", "stuid", is_unique=1),
                profiled_row("has_pet", "stuid"),
            ],
        )
        result = add_serving_cross_table_features(
            df_final,
            discovered(("student", "stuid"), ("has_pet", "stuid")),
            FakeUniquenessEngine(),
        )
        parent = result[result.table_name == "student"].iloc[0]

        assert parent["is_non_unique_and_name_unique_elsewhere"] == 0


class TestStreamingRunScope:
    """A run narrowed to changed tables must still see the whole schema."""

    @pytest.mark.parametrize(
        "_stub_read_sql",
        [
            [
                {
                    "database": "iceberg",
                    "schema": "new_predict_data",
                    "table_name": "student",
                    "column_name": "stuid",
                    "is_unique": 1,
                    "created_at": "2026-08-01 10:00:00",
                },
            ],
        ],
        indirect=True,
    )
    def test_parent_outside_the_run_is_found_via_the_queue(self):
        """Only `has_pet` was reprofiled; `student.stuid` is known only from the queue."""
        df_final = pd.DataFrame([profiled_row("has_pet", "stuid")])

        result = add_serving_cross_table_features(
            df_final,
            discovered(("student", "stuid"), ("has_pet", "stuid")),
            FakeUniquenessEngine(),
        )

        assert len(result) == 1
        child = result.iloc[0]
        assert child["name_unique_in_other_table"] == 1, (
            "the parent is in the queue but was not used as reference"
        )
        assert child["n_other_tables_with_same_column_name"] == 1

    def test_without_queue_history_the_parent_is_not_invented(self):
        """No recorded uniqueness means no parent - the safe direction to be wrong in."""
        df_final = pd.DataFrame([profiled_row("has_pet", "stuid")])

        result = add_serving_cross_table_features(
            df_final,
            discovered(("student", "stuid"), ("has_pet", "stuid")),
            FakeUniquenessEngine(),
        )
        child = result.iloc[0]

        assert child["name_unique_in_other_table"] == 0
        # The name is still visible even though the uniqueness is not.
        assert child["n_other_tables_with_same_column_name"] == 1

    def test_a_missing_queue_table_is_not_fatal(self):
        """First ever run: the queue does not exist yet, and profiling must still finish."""
        df_final = pd.DataFrame([profiled_row("has_pet", "stuid")])

        result = add_serving_cross_table_features(
            df_final,
            discovered(("has_pet", "stuid")),
            FakeUniquenessEngine(fail=True),
        )

        assert len(result) == 1
        assert set(CROSS_TABLE_FEATURES) <= set(result.columns)


class TestFrameContract:
    def test_only_the_profiled_rows_come_back_in_their_original_order(self):
        df_final = pd.DataFrame(
            [
                profiled_row("has_pet", "petid", ordinal=1),
                profiled_row("has_pet", "stuid", ordinal=2),
            ],
        )
        result = add_serving_cross_table_features(
            df_final,
            discovered(
                ("student", "stuid"),
                ("teacher", "tid"),
                ("has_pet", "petid"),
                ("has_pet", "stuid"),
            ),
            FakeUniquenessEngine(),
        )

        assert len(result) == 2
        assert result["column_name"].tolist() == ["petid", "stuid"]
        assert result["ordinal_position"].tolist() == [1, 2]

    def test_features_are_integers_for_the_bigint_queue_columns(self):
        """A float column here is rejected by the column-explicit INSERT into the queue."""
        df_final = pd.DataFrame([profiled_row("has_pet", "stuid")])
        result = add_serving_cross_table_features(
            df_final,
            discovered(("student", "stuid"), ("has_pet", "stuid")),
            FakeUniquenessEngine(),
        )

        for feature in CROSS_TABLE_FEATURES:
            assert result[feature].dtype.kind == "i", f"{feature} is not an integer column"

    def test_original_columns_survive(self):
        df_final = pd.DataFrame([profiled_row("has_pet", "stuid")])
        result = add_serving_cross_table_features(
            df_final,
            discovered(("has_pet", "stuid")),
            FakeUniquenessEngine(),
        )

        assert set(df_final.columns) <= set(result.columns)
        assert set(CROSS_TABLE_FEATURES) <= set(result.columns)

    def test_reference_rows_never_leak_into_the_result(self):
        """`teacher` was never profiled, so it must not appear as a row to be predicted."""
        df_final = pd.DataFrame([profiled_row("has_pet", "stuid")])
        result = add_serving_cross_table_features(
            df_final,
            discovered(("teacher", "tid"), ("has_pet", "stuid")),
            FakeUniquenessEngine(),
        )

        assert result["table_name"].tolist() == ["has_pet"]
