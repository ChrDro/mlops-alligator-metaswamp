"""
Tests for the TPC-H recipes (Phase 2b of TASK_3_PLAN.md).

The centrepiece is ``test_every_recipe_lands_on_its_intended_normal_form``: each recipe
was designed for one label, and the FD derivation through the join ladder is fiddly
enough that "it still compiles" says nothing. If a join step or a projection changes, that
test is what notices the label moved with it.

No database is touched. The SQL is asserted structurally, never executed.
"""

import pytest
from nf_generator import label_of
from nf_labeling import closure, normal_form, normalise_fds
from nf_recipes import (
    JOIN_RECIPES,
    LIST_RECIPES,
    REPEATING_RECIPES,
    TPCH,
    JoinStep,
    build_join,
    build_specs,
    obfuscate,
    project_fds,
)


# What each recipe was built to produce. Written down here rather than derived, so a
# change in the FD algebra shows up as a failing expectation instead of a moving target.
EXPECTED_LABELS = {
    # 3NF, single key
    "orders_core": 3,
    "customer_core": 3,
    "supplier_core": 3,
    "part_core": 3,
    "nation_core": 3,
    # 3NF, composite key - the anti-leak family
    "lineitem_core": 3,
    "partsupp_core": 3,
    # 2NF, single key
    "orders_customer": 2,
    "orders_customer_nation": 2,
    "orders_customer_nation_region": 2,
    "customer_nation": 2,
    "supplier_nation": 2,
    "part_with_mfgr": 2,
    # 2NF, composite key
    "lineitem_with_status": 2,
    "lineitem_part": 2,
    "lineitem_supplier_nation": 2,
    # 1NF, partial dependencies
    "lineitem_orders": 1,
    "lineitem_orders_customer": 1,
    "partsupp_supplier": 1,
    "partsupp_part": 1,
    "partsupp_part_supplier": 1,
}


@pytest.mark.parametrize("recipe", JOIN_RECIPES, ids=lambda r: r.recipe_id)
def test_every_recipe_lands_on_its_intended_normal_form(recipe):
    relation = recipe.build("tiny", 2000)

    assert (
        normal_form(relation.attributes, relation.fds, violates_1nf=False)
        == (EXPECTED_LABELS[recipe.recipe_id])
    )


def test_expectations_cover_every_recipe():
    """A new recipe without an expected label would otherwise be silently untested."""
    assert {recipe.recipe_id for recipe in JOIN_RECIPES} == set(EXPECTED_LABELS)


# --------------------------------------------------------------------------------------
# FD algebra
# --------------------------------------------------------------------------------------


def test_projection_keeps_a_dependency_whose_middle_attribute_is_dropped():
    """
    The one that filtering cannot get right.

    With ``a -> b`` and ``b -> c``, dropping ``b`` leaves ``a -> c`` standing. Discarding
    both FDs because they mention ``b`` would make the projection look better normalized
    than it is - and that is the direction of error nobody questions.
    """
    fds = normalise_fds([({"a"}, {"b"}), ({"b"}, {"c"})])

    projected = project_fds(fds, keep=["a", "c"])

    assert closure({"a"}, projected) == {"a", "c"}


def test_projection_drops_a_dependency_whose_dependent_is_gone():
    fds = normalise_fds([({"a"}, {"b"})])

    assert project_fds(fds, keep=["a"]) == []


# --------------------------------------------------------------------------------------
# Joins
# --------------------------------------------------------------------------------------


def test_join_makes_the_foreign_key_determine_the_dimension():
    relation = build_join("orders", steps=(JoinStep("custkey", "customer"),))

    assert "customer_mktsegment" in relation.attributes
    # The dimension's own key column is not projected a second time.
    assert "customer_custkey" not in relation.attributes
    assert closure({"custkey"}, relation.fds) >= {"customer_mktsegment", "customer_nationkey"}


def test_chained_join_resolves_against_the_right_alias():
    """``nation`` hangs off ``customer``, not off the fact table."""
    relation = build_join(
        "orders",
        steps=(JoinStep("custkey", "customer"), JoinStep("customer_nationkey", "nation")),
    )

    assert 't1."nationkey" = t2."nationkey"' in relation.sql
    assert closure({"custkey"}, relation.fds) >= {"nation_name", "nation_regionkey"}


def test_alternate_key_inside_a_dimension_survives_the_join():
    """``nation.name -> nationkey`` still holds once nation has been folded in."""
    relation = build_join("customer", steps=(JoinStep("nationkey", "nation"),))

    assert closure({"nation_name"}, relation.fds) >= {"nationkey", "nation_regionkey"}


def test_join_step_with_an_unknown_foreign_key_is_rejected():
    with pytest.raises(ValueError, match="not in the relation yet"):
        build_join("orders", steps=(JoinStep("nationkey", "nation"),))


def test_row_limit_orders_by_the_key_in_declaration_order():
    """
    Sorting the key alphabetically was measured to break the sample.

    For lineitem it produces ``ORDER BY linenumber, orderkey``, and the first 2000 rows of
    that ordering all carry linenumber = 1: the column goes constant, orderkey becomes
    unique, and the composite key the recipe exists for is gone from the data.
    """
    relation = build_join("lineitem", limit=2000)

    assert 'ORDER BY "orderkey", "linenumber" LIMIT 2000' in relation.sql


# --------------------------------------------------------------------------------------
# 0NF injections and their controls
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("recipe", LIST_RECIPES, ids=lambda r: r.recipe_id)
def test_injection_and_control_differ_only_in_the_last_column(recipe):
    """
    The matched pair is the whole design.

    Same query, same grain, same column count - if anything else differed, the model could
    separate 0NF from its control on that instead of on atomicity, which is exactly how
    finding 1.3 happened.
    """
    injected = recipe.build("tiny", 2000, atomic=False)
    control = recipe.build("tiny", 2000, atomic=True)

    assert len(injected.attributes) == len(control.attributes)
    assert injected.attributes[:-1] == control.attributes[:-1]
    assert injected.attributes[-1] != control.attributes[-1]
    assert recipe.separator in injected.sql


@pytest.mark.parametrize("recipe", REPEATING_RECIPES, ids=lambda r: r.recipe_id)
def test_repeating_group_and_control_have_the_same_shape(recipe):
    injected = recipe.build("tiny", 2000, atomic=False)
    control = recipe.build("tiny", 2000, atomic=True)

    assert len(injected.attributes) == len(control.attributes)
    # Numbered columns in the injection, none in the control.
    assert [name for name in injected.attributes if name[-1].isdigit()]
    assert not [name for name in control.attributes if name[-1].isdigit()]


def test_control_of_a_two_nf_parent_stays_two_nf():
    """0NF must not correlate with the parent's normal form."""
    recipe = next(r for r in LIST_RECIPES if r.parent == "orders_customer")
    control = recipe.build("tiny", 2000, atomic=True)

    assert normal_form(control.attributes, control.fds, violates_1nf=False) == 2


# --------------------------------------------------------------------------------------
# Variation axes
# --------------------------------------------------------------------------------------


def test_obfuscated_names_contain_no_digits():
    """
    Numbered obfuscation (``col_01``, ``col_02``) is exactly the repeating-group pattern.

    It would turn every obfuscated table into a fake 1NF violation - a generator artifact
    of precisely the kind Phase 2 exists to remove.
    """
    mapping = obfuscate([f"attribute_{i}" for i in range(30)])

    assert all(not any(character.isdigit() for character in name) for name in mapping.values())
    assert len(set(mapping.values())) == 30


def test_table_names_carry_no_hint_of_the_label():
    """
    The extractor sees table names.

    A name like ``tbl_dirty_1nf_7`` is a label the model can read straight off the
    catalog, which is why the names here are positional.
    """
    forbidden = ("1nf", "2nf", "3nf", "0nf", "dirty", "denorm", "partial", "transitive")

    for spec in build_specs():
        lowered = spec.table_name.lower()
        assert not any(token in lowered for token in forbidden), spec.table_name


def test_the_generated_set_covers_all_four_classes():
    labels = {label_of(spec) for spec in build_specs()}

    assert labels == {0, 1, 2, 3}


def test_every_injection_is_paired_with_a_control():
    """An unpaired injection would leave the atomicity claim unmeasurable."""
    pairs: dict[str, set[str]] = {}
    for spec in build_specs():
        pair_id = spec.generation_params.get("pair_id")
        if pair_id:
            pairs.setdefault(pair_id, set()).add(spec.generation_params["kind"])

    assert pairs
    assert all(kinds == {"injection", "control"} for kinds in pairs.values())


def test_declared_attributes_match_the_select_list():
    """
    ``nf_generator._verify_columns`` checks this against the database; this checks it
    against the SQL, so a mismatch fails in the test suite instead of mid-generation.
    """
    for spec in build_specs():
        for attribute in spec.attributes:
            assert f'"{attribute}"' in spec.source_sql, (spec.table_name, attribute)


def test_base_relations_only_declare_attributes_they_have():
    for relation in TPCH.values():
        for lhs, rhs in relation.fds():
            assert (lhs | rhs) <= set(relation.attributes), relation.name
