"""
Test oracle for the normal-form label (Phase 2a of TASK_3_PLAN.md).

Every expected value below was worked out by hand, not read off the implementation.
That is the whole point: this function produces the **y** of the next training set, so a
bug here does not show up as a crash but as a quietly mislabeled dataset - exactly the
failure mode that made the current data unusable (findings 1.1-1.3).

The eight oracle cases cover the four places the computation is usually gotten wrong:
several candidate keys, a partial dependency reachable only transitively, the "dependent
is prime" escape hatch of 3NF, and 1NF not being derivable from FDs at all.
"""

import pytest
from nf_labeling import (
    MAX_SEARCHED_ATTRIBUTES,
    analyze,
    candidate_keys,
    closure,
    normal_form,
    normalise_fds,
    prime_attributes,
)


# --------------------------------------------------------------------------------------
# The oracle: (id, attributes, fds, violates_1nf, expected label)
# --------------------------------------------------------------------------------------

# 3NF. Single-attribute key, everything hangs off it directly.
SIMPLE_KEY = (
    ["order_id", "order_date", "total"],
    [({"order_id"}, {"order_date", "total"})],
)

# 1NF. Classic enrolment table: student_name depends on half of the composite key.
PARTIAL_DEPENDENCY = (
    ["student_id", "course_id", "grade", "student_name"],
    [({"student_id", "course_id"}, {"grade"}), ({"student_id"}, {"student_name"})],
)

# 2NF. Fully dependent on the key, but department_head hangs off department_id.
TRANSITIVE_CHAIN = (
    ["employee_id", "department_id", "department_head"],
    [({"employee_id"}, {"department_id"}), ({"department_id"}, {"department_head"})],
)

# 3NF with two candidate keys, {a} and {b}. Taking only the first would mark b
# non-prime and mislabel the table.
TWO_CANDIDATE_KEYS = (
    ["a", "b", "c"],
    [({"a"}, {"b", "c"}), ({"b"}, {"a", "c"})],
)

# 3NF, and the case that separates a correct implementation from a plausible one.
# Keys are {city, street} and {street, zip}, so *every* attribute is prime. zip -> city
# has a non-superkey determinant, yet the table is in 3NF because city is prime.
PRIME_ON_THE_RIGHT = (
    ["city", "street", "zip"],
    [({"city", "street"}, {"zip"}), ({"zip"}, {"city"})],
)

# 3NF. No FDs at all -> the key is the full attribute set, so nothing is non-prime and
# there is nothing left that could violate anything.
NO_DEPENDENCIES = (["a", "b", "c"], [])

# 1NF. Partial *and* transitive violations at once - 2NF is checked first, so the label
# must be 1, not 2.
PARTIAL_AND_TRANSITIVE = (
    ["student_id", "course_id", "grade", "advisor", "advisor_office"],
    [
        ({"student_id", "course_id"}, {"grade"}),
        ({"student_id"}, {"advisor"}),
        ({"advisor"}, {"advisor_office"}),
    ],
)

# Edge case: a one-column relation with an empty FD set. The key is the column itself.
SINGLE_ATTRIBUTE = (["id"], [])


ORACLE = [
    ("simple key", *SIMPLE_KEY, False, 3),
    ("partial dependency", *PARTIAL_DEPENDENCY, False, 1),
    ("transitive chain", *TRANSITIVE_CHAIN, False, 2),
    ("two candidate keys", *TWO_CANDIDATE_KEYS, False, 3),
    ("prime attribute on the right", *PRIME_ON_THE_RIGHT, False, 3),
    ("no dependencies", *NO_DEPENDENCIES, False, 3),
    ("partial beats transitive", *PARTIAL_AND_TRANSITIVE, False, 1),
    ("single attribute", *SINGLE_ATTRIBUTE, False, 3),
    # 1NF is measured, not derived: the same 3NF schema drops to 0 on non-atomic values.
    ("non-atomic values", *PRIME_ON_THE_RIGHT, True, 0),
]


@pytest.mark.parametrize(("case_id", "attributes", "fds", "violates_1nf", "expected"), ORACLE)
def test_normal_form_matches_hand_computed_label(case_id, attributes, fds, violates_1nf, expected):
    assert normal_form(attributes, fds, violates_1nf) == expected, case_id


@pytest.mark.parametrize(("case_id", "attributes", "fds", "violates_1nf", "expected"), ORACLE)
def test_analysis_evidence_agrees_with_the_label(case_id, attributes, fds, violates_1nf, expected):
    """
    The label and the recorded violations must not contradict each other.

    A label of 2 while ``partial_dependencies`` is non-empty would mean the manifest
    tells the validation harness (2c) a different story than the training data does.
    """
    result = analyze(attributes, fds, violates_1nf)

    assert result.normal_form == expected, case_id
    assert result.reason, case_id
    if result.normal_form >= 2:
        assert result.partial_dependencies == (), case_id
    if result.normal_form == 3:
        assert result.transitive_dependencies == (), case_id
    if result.normal_form == 1:
        assert result.partial_dependencies, case_id


def test_violates_1nf_outranks_a_clean_fd_set():
    """Atomicity is a property of the values; no FD set can rescue or produce it."""
    attributes, fds = SIMPLE_KEY

    assert normal_form(attributes, fds, violates_1nf=False) == 3
    assert normal_form(attributes, fds, violates_1nf=True) == 0


# --------------------------------------------------------------------------------------
# closure
# --------------------------------------------------------------------------------------


def test_closure_follows_chains():
    fds = normalise_fds([({"a"}, {"b"}), ({"b"}, {"c"}), ({"c"}, {"d"})])

    assert closure({"a"}, fds) == {"a", "b", "c", "d"}
    assert closure({"c"}, fds) == {"c", "d"}


def test_closure_needs_the_full_left_hand_side():
    fds = normalise_fds([({"a", "b"}, {"c"})])

    assert closure({"a"}, fds) == {"a"}
    assert closure({"a", "b"}, fds) == {"a", "b", "c"}


def test_closure_of_the_empty_set_is_empty_without_constants():
    assert closure(set(), normalise_fds([({"a"}, {"b"})])) == frozenset()


# --------------------------------------------------------------------------------------
# candidate_keys / prime attributes
# --------------------------------------------------------------------------------------


def test_candidate_keys_returns_all_of_them():
    attributes, fds = TWO_CANDIDATE_KEYS

    keys = candidate_keys(attributes, normalise_fds(fds))

    assert sorted(keys, key=sorted) == [frozenset({"a"}), frozenset({"b"})]


def test_prime_is_the_union_over_all_keys():
    """With keys {city, street} and {street, zip}, every attribute is prime."""
    attributes, fds = PRIME_ON_THE_RIGHT

    assert prime_attributes(attributes, normalise_fds(fds)) == {"city", "street", "zip"}


def test_keys_are_minimal():
    attributes, fds = PARTIAL_DEPENDENCY

    keys = candidate_keys(attributes, normalise_fds(fds))

    assert keys == [frozenset({"student_id", "course_id"})]


def test_attribute_on_no_right_hand_side_is_in_every_key():
    """`c` is determined by nothing, so no key can leave it out."""
    keys = candidate_keys(["a", "b", "c"], normalise_fds([({"a"}, {"b"})]))

    assert keys == [frozenset({"a", "c"})]


def test_attribute_only_on_right_hand_sides_is_in_no_key():
    """`b` derives nothing and is itself derivable - it cannot be part of a minimal key."""
    keys = candidate_keys(["a", "b", "c"], normalise_fds([({"a"}, {"b"})]))

    assert all("b" not in key for key in keys)


def test_key_search_refuses_an_intractable_fd_set():
    """
    The guard, not the search, must decide here.

    21 attributes in a dependency cycle plus one that nothing determines: the cycle
    leaves every one of the 21 undecided, which is one over the limit. Without the
    guard this call would enumerate 2**21 subsets.
    """
    cycle = [f"a{i}" for i in range(MAX_SEARCHED_ATTRIBUTES + 1)]
    fds = [({name}, {cycle[(i + 1) % len(cycle)]}) for i, name in enumerate(cycle)]

    with pytest.raises(ValueError, match="key search is exponential"):
        candidate_keys([*cycle, "standalone"], normalise_fds(fds))


def test_empty_relation_is_rejected():
    with pytest.raises(ValueError, match="at least one attribute"):
        candidate_keys([], [])


# --------------------------------------------------------------------------------------
# normalise_fds - the guard against typos in generator specs
# --------------------------------------------------------------------------------------


def test_unknown_attribute_in_an_fd_is_an_error():
    """
    A typo must fail loudly.

    Silently ignoring `stundent_id` would drop the partial dependency and label the
    table 3NF - a wrong label that nothing downstream could detect.
    """
    with pytest.raises(ValueError, match="unknown attributes"):
        normalise_fds([({"stundent_id"}, {"grade"})], attributes=["student_id", "grade"])


def test_trivial_parts_are_removed():
    """`{a, b} -> {b, c}` carries only `{a, b} -> {c}`."""
    assert normalise_fds([({"a", "b"}, {"b", "c"})]) == [(frozenset({"a", "b"}), frozenset({"c"}))]


def test_fully_trivial_fd_is_dropped_entirely():
    assert normalise_fds([({"a", "b"}, {"a"})]) == []


# --------------------------------------------------------------------------------------
# Documented convention, not a law of database theory
# --------------------------------------------------------------------------------------


def test_constant_column_counts_as_a_partial_dependency():
    """
    An FD with an empty left-hand side means the column is constant.

    The empty set is a proper subset of every non-empty key, so a constant non-prime
    column is treated as a 2NF violation. Defensible, but a convention - if the
    generator ever emits constant columns on purpose, revisit this together with the
    ``_partial_dependencies`` loop rather than being surprised by the label.
    """
    result = analyze(["id", "tenant"], [(set(), {"tenant"})], violates_1nf=False)

    assert result.normal_form == 1
    assert result.partial_dependencies == ((frozenset(), "tenant"),)
