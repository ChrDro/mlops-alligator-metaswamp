"""
Tests for the generator skeleton (Phase 2a of TASK_3_PLAN.md).

Two things are worth testing before the recipes exist (Phase 2b): that the label a spec
gets comes from its declared FDs, and that the two ways a recipe can quietly produce a
wrong label are caught - a table name that is not an identifier (it goes into DDL, where
it cannot be a bound parameter) and a ``source_sql`` whose columns do not match the
declared attributes.
"""

import nf_generator
import pytest
from nf_generator import TableSpec, build_row, generate, materialise

from .conftest import FakeConnection


ENROLMENT = TableSpec(
    table_name="tbl_enrolment_2nf_violation",
    source_sql="SELECT 1 AS student_id, 2 AS course_id, 'A' AS grade, 'x' AS student_name",
    attributes=("student_id", "course_id", "grade", "student_name"),
    declared_fds=(
        (frozenset({"student_id", "course_id"}), frozenset({"grade"})),
        (frozenset({"student_id"}), frozenset({"student_name"})),
    ),
    generation_params={"rows": 5000},
)

# What information_schema returns for the table above.
ENROLMENT_COLUMNS = [("student_id",), ("course_id",), ("grade",), ("student_name",)]


def test_label_comes_from_the_declared_fds():
    assert build_row(ENROLMENT).target_normal_form == 1


def test_violates_1nf_reaches_the_label():
    """The recipe has to state it - no FD set can express non-atomic values."""
    spec = TableSpec(
        table_name="tbl_enrolment_flattened",
        source_sql=ENROLMENT.source_sql,
        attributes=ENROLMENT.attributes,
        declared_fds=ENROLMENT.declared_fds,
        violates_1nf=True,
    )

    row = build_row(spec)

    assert row.target_normal_form == 0
    assert row.violates_1nf is True


def test_materialise_replaces_instead_of_appending():
    """Regeneration must be repeatable; an appended-to table breaks its own FDs."""
    conn = FakeConnection(rows=ENROLMENT_COLUMNS)

    materialise(conn, ENROLMENT)

    assert conn.statements("DROP TABLE IF EXISTS")
    create = conn.statements("CREATE TABLE iceberg.nf_training.")[0]
    assert create.endswith(ENROLMENT.source_sql)


def test_column_mismatch_is_an_error():
    """
    The failure this guard exists for is silent.

    A renamed column makes its FD never fire, which makes the table look *better*
    normalized than it is - so the label is wrong in the direction nothing downstream
    would question.
    """
    conn = FakeConnection(rows=[("student_id",), ("course_id",), ("grade",), ("name",)])

    with pytest.raises(ValueError, match="do not match the declared attributes"):
        materialise(conn, ENROLMENT)


def test_table_name_must_be_an_identifier():
    """It is interpolated into DDL, where a bound parameter is not an option."""
    spec = TableSpec(
        table_name="tbl; DROP TABLE iceberg.predictions.queue",
        source_sql="SELECT 1 AS a",
        attributes=("a",),
    )
    conn = FakeConnection(rows=[("a",)])

    with pytest.raises(ValueError, match="not a plain SQL identifier"):
        materialise(conn, spec)


def test_labels_are_computed_before_anything_is_written():
    """
    A broken FD declaration must fail on the first spec, not halfway through a rebuild.

    The spec below names an attribute the relation does not have, so ``generate`` has to
    raise without having issued a single statement.
    """
    broken = TableSpec(
        table_name="tbl_broken",
        source_sql="SELECT 1 AS a",
        attributes=("a", "b"),
        declared_fds=((frozenset({"a"}), frozenset({"typo"})),),
    )
    conn = FakeConnection(rows=ENROLMENT_COLUMNS)

    with pytest.raises(ValueError, match="unknown attributes"):
        generate(conn, [ENROLMENT, broken])

    assert conn.executed == []


def test_recipes_are_loaded_lazily_and_produce_specs():
    """
    ``load_specs`` imports ``nf_recipes`` inside the function, not at module level.

    The recipes need ``TableSpec`` from the generator, so a top-level import in both
    directions would be a cycle. This checks the seam actually holds - a stray top-level
    import would make the module fail to load rather than fail here.
    """
    specs = nf_generator.load_specs()

    assert specs
    assert all(spec.attributes for spec in specs)
    assert len({spec.table_name for spec in specs}) == len(specs)
