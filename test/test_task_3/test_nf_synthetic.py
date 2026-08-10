"""
Tests for the synthetic schema generator (Phase 2d of TASK_3_PLAN.md).

The generator's job is to supply what TPC-H cannot: the *number* of independent source
schemas. Two things therefore have to hold, and both are checked here rather than hoped
for - that each schema really carries the normal form it was asked for, and that none of
the variation axes leaks the label.

No database is touched. The SQL is asserted structurally, never executed; whether the
declared dependencies survive materialization is what the validation harness (2c) measures
against the real tables.
"""

import collections

import pytest
from nf_generator import label_of
from nf_labeling import normal_form
from nf_synthetic import (
    COLUMN_COUNTS,
    NULL_RATES,
    ROW_COUNTS,
    build_schema,
    build_synthetic_specs,
)


SPECS = build_synthetic_specs(300)


# --------------------------------------------------------------------------------------
# The construction produces the normal form it claims
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("target", [0, 1, 2, 3])
def test_a_schema_carries_the_normal_form_it_was_built_for(target):
    schema = build_schema(
        schema_id=1,
        target=target,
        n_attributes=10,
        key_size=2,
        n_rows=1000,
        null_rate=0.0,
        naming="real",
        base=2 if target == 0 else None,
    )

    assert normal_form(schema.attributes, schema.fds, schema.violates_1nf) == target


def test_every_generated_spec_matches_its_label():
    """
    ``build_schema`` verifies its own output, so this is the guard on the *sweep*: a
    combination of axes that quietly produces a different structure than intended.
    """
    labels = collections.Counter(label_of(spec) for spec in SPECS)
    assert set(labels) == {0, 1, 2, 3}
    assert min(labels.values()) >= len(SPECS) // 8


def test_a_partial_dependency_needs_a_composite_key():
    """1NF is defined by one, so asking for it with a single-column key is a mistake."""
    with pytest.raises(ValueError, match="composite key"):
        build_schema(
            schema_id=1,
            target=1,
            n_attributes=10,
            key_size=1,
            n_rows=1000,
            null_rate=0.0,
            naming="real",
        )


def test_a_structure_that_does_not_fit_is_rejected():
    with pytest.raises(ValueError, match="non-key attributes"):
        build_schema(
            schema_id=1,
            target=1,
            n_attributes=4,
            key_size=2,
            n_rows=1000,
            null_rate=0.0,
            naming="real",
        )


# --------------------------------------------------------------------------------------
# None of the variation axes may give the label away
# --------------------------------------------------------------------------------------


def test_column_count_does_not_give_the_label_away():
    """
    The 0NF list column replaces an attribute instead of joining it.

    Added on top, every 0NF table would have one column more than the axis says - 6/11/21/41
    against 5/10/20/40 - and the column count alone would be a perfect label hint.
    """
    widths = {len(spec.attributes) for spec in SPECS}

    assert widths == set(COLUMN_COUNTS)
    for label in (0, 1, 2, 3):
        assert {len(s.attributes) for s in SPECS if label_of(s) == label} == set(COLUMN_COUNTS)


@pytest.mark.parametrize(
    ("axis", "expected"),
    [("row_count", set(ROW_COUNTS)), ("null_rate", set(NULL_RATES))],
)
def test_every_class_spans_every_axis(axis, expected):
    """
    If an axis and the label moved together, the axis would become a label hint - which is
    exactly how finding 1.3 happened with `table_avg_unique_ratio`.
    """
    for label in (0, 1, 2, 3):
        seen = {spec.generation_params[axis] for spec in SPECS if label_of(spec) == label}
        assert seen == expected, label


def test_zero_nf_is_built_on_every_underlying_structure():
    """
    Atomicity is independent of the dependency structure, and the data has to say so.

    If every 0NF table also carried a partial dependency, "not atomic" and "has a partial
    dependency" would always travel together and the model could read one off the other.
    """
    bases = {spec.generation_params["base"] for spec in SPECS if label_of(spec) == 0}

    assert bases == {1, 2, 3}


def test_each_schema_is_its_own_group():
    """
    The point of 2d: 300 *source schemas*, not 300 tables from a handful of shapes.

    Grouping them by structural family instead would put 300 schemas into 16 groups and
    hand the split back the very problem it exists to measure.
    """
    schemas = {spec.generation_params["recipe_id"] for spec in SPECS}
    families = {spec.generation_params["family"] for spec in SPECS}

    assert len(schemas) == len(SPECS)
    assert len(families) < len(schemas)


def test_table_names_carry_no_hint_of_the_label():
    forbidden = ("0nf", "1nf", "2nf", "3nf", "partial", "transitive", "dirty")

    for spec in SPECS:
        assert not any(token in spec.table_name.lower() for token in forbidden)


def test_obfuscated_column_names_contain_no_digits():
    """A numeric suffix is the repeating-group pattern - it would fake a 1NF violation."""
    obfuscated = [s for s in SPECS if s.generation_params["naming"] == "obfuscated"]

    assert obfuscated
    for spec in obfuscated:
        for name in spec.attributes:
            assert not any(character.isdigit() for character in name), name


# --------------------------------------------------------------------------------------
# The SQL
# --------------------------------------------------------------------------------------


def test_a_dependent_column_repeats_its_determinant_expression():
    """
    Trino cannot reference an output alias from elsewhere in the same SELECT list.

    Inlining the determinant's expression is also what keeps the dependency exact: the
    dependent is a function of the very same value, not of a column that happens to look
    like it.
    """
    schema = build_schema(
        schema_id=3,
        target=2,
        n_attributes=8,
        key_size=1,
        n_rows=1000,
        null_rate=0.0,
        naming="obfuscated",
    )
    determinant = next(
        sorted(lhs)[0] for lhs, _ in schema.fds if len(lhs) == 1 and "x" in sorted(lhs)[0]
    )

    assert f'"{determinant}"' not in schema.sql.split("AS")[0]


def _select_line(sql: str, column: str) -> str:
    """The one line of the SELECT list that produces ``column`` - one column per line."""
    suffix = f'AS "{column}"'
    line = next(line for line in sql.splitlines() if line.strip().rstrip(",").endswith(suffix))
    return line.strip().removeprefix("SELECT ")


@pytest.mark.parametrize("target", [1, 2])
def test_a_dependent_column_goes_null_by_determinant_not_by_row(target):
    """
    A dependent column nulled per row breaks the FD it was built to carry.

    Two rows sharing a determinant value would get independent draws - one keeps its
    value, the other goes NULL - so the determinant no longer maps to a single image. The
    whole FD group has to go NULL together. Found by the label check at 432-table scale
    after the fact; this pins it at 0.2 seconds.
    """
    schema = build_schema(
        schema_id=7,
        target=target,
        n_attributes=10,
        key_size=2,
        n_rows=1000,
        null_rate=0.30,
        naming="convention",
    )
    everything = frozenset(schema.attributes)
    # Key-style FDs (the key itself, a surrogate) determine every other attribute and
    # their dependents include ungated columns - the gating rule applies to the chain
    # and partial FDs, whose determinant may be one column or (since 2026-08-07) a pair.
    narrow_fds = [(lhs, rhs) for lhs, rhs in schema.fds if rhs != everything - lhs]
    assert narrow_fds, "a 1NF/2NF schema must declare at least one dependency"

    for lhs, rhs in narrow_fds:
        for dependent in rhs:
            line = _select_line(schema.sql, dependent)
            gate = line[: line.index("THEN NULL")]
            for determinant in lhs:
                determinant_expression = _select_line(schema.sql, determinant).rsplit(" AS ", 1)[0]
                assert determinant_expression.strip() in gate, (determinant, dependent)


def test_the_dependency_count_does_not_give_the_label_away():
    """
    Counting dependencies must not read off the class.

    With exactly one transitive dependency for 2NF and one of each for 1NF, the plain count
    was a fingerprint: ``table_fd_count`` alone scored 0.6655 weighted F1 against a 0.6
    limit and the signature check in 2c failed on it. Only the *position* of a determinant
    may carry the distinction, never how many there are.

    Key-style FDs are excluded here for the same reason ``find_dependencies`` drops them:
    a determinant that is unique determines everything, which is no finding. That covers
    the key's own FD and (since 2026-08-07) the surrogate-key decoy, which declares a
    second key -> everything dependency on schemas of every class.
    """
    counts = collections.defaultdict(set)
    for spec in SPECS:
        label = spec.generation_params["family"].split("_")[1]
        everything = frozenset(spec.attributes)
        counts[label].add(
            sum(1 for lhs, rhs in spec.declared_fds if len(lhs) == 1 and rhs != everything - lhs),
        )

    shared = counts["1nf"] & counts["2nf"]
    assert len(shared) >= 3, f"1NF {sorted(counts['1nf'])} vs 2NF {sorted(counts['2nf'])}"
    assert counts["3nf"] == {0}, "a 3NF relation has no dependency off a non-key determinant"


def test_declared_attributes_match_the_select_list():
    for spec in SPECS:
        for attribute in spec.attributes:
            assert f'"{attribute}"' in spec.source_sql, (spec.table_name, attribute)


def test_generation_is_reproducible():
    """
    Regenerating has to produce the same data, or 2c compares the label against a table
    that no longer exists.
    """
    assert [s.source_sql for s in build_synthetic_specs(20)] == [
        s.source_sql for s in build_synthetic_specs(20)
    ]
