"""
Tests for the shared normal-form feature extractor (Phase 2b of TASK_3_PLAN.md).

``nf_features`` is the module that has to be right in two places at once - the training
set is built with it and the Prefect pipeline serves with it, so a change here moves
training and serving together instead of pulling them apart (finding 1.6).

Everything below is tested without a database. The measurements are handed in as a
``TableProfile``; what is under test is the reasoning on top of them.
"""

from itertools import combinations

import nf_features
import pytest
from nf_features import (
    FEATURE_COLUMNS,
    FLOAT_FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    LIST_TOKEN_MAX_MEAN_LENGTH,
    LIST_VALUE_MIN_SHARE,
    MAX_LEVEL_3_CANDIDATES,
    MAX_PROFILED_COLUMNS,
    SAMPLE_ROW_THRESHOLD,
    SAMPLE_TARGET_ROWS,
    ColumnInfo,
    TableProfile,
    _sampled_source,
    classify_dependencies,
    count_distinct_combinations,
    dependency_strength,
    find_dependencies,
    find_near_dependencies,
    find_near_uccs,
    find_uccs,
    is_freetext_name,
    profile_table,
    repeating_group_columns,
    ucc_score,
)
from sqlalchemy.exc import OperationalError

from .conftest import FakeConnection


def _profile(row_count, distinct, pair_distinct=None, **kwargs):  # noqa: ANN003
    """A TableProfile with only the fields a given test cares about filled in."""
    names = list(distinct)
    zeros = dict.fromkeys(names, 0.0)
    return TableProfile(
        row_count=row_count,
        columns=tuple(ColumnInfo(name=name, data_type="varchar") for name in names),
        distinct=distinct,
        non_null=kwargs.get("non_null", dict.fromkeys(names, row_count)),
        list_like_ratio=kwargs.get("list_like_ratio", zeros),
        mean_length=kwargs.get("mean_length", zeros),
        mean_separators=kwargs.get("mean_separators", zeros),
        pair_distinct=pair_distinct or {},
    )


# --------------------------------------------------------------------------------------
# Atomicity by name: repeating groups
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("columns", "expected"),
    [
        (["id", "phone1", "phone2", "phone3"], {"phone1", "phone2", "phone3"}),
        (["id", "order_1", "order_2"], {"order_1", "order_2"}),
        (["id", "phone1"], set()),  # a single numbered column is not a group
        (["id", "name", "city"], set()),
    ],
)
def test_repeating_groups_are_found_by_name(columns, expected):
    """The 1NF violation no value statistic can see - the values are perfectly atomic."""
    assert repeating_group_columns(columns) == expected


def test_a_table_wide_naming_scheme_is_not_a_repeating_group():
    """
    ``col_01`` .. ``col_12`` is obfuscated naming, not a flattened 1:n relationship.

    Without this guard, every table with positional column names would be labelled 0NF by
    a feature - a generator artifact of exactly the kind Phase 2 exists to remove.
    """
    assert repeating_group_columns([f"col_{i:02d}" for i in range(1, 13)]) == set()


def test_freetext_names_are_recognised():
    assert is_freetext_name("bemerkung")
    assert is_freetext_name("customer_comment")
    assert not is_freetext_name("customer_id")


# --------------------------------------------------------------------------------------
# Dependency discovery
# --------------------------------------------------------------------------------------


def test_dependency_holds_when_the_pair_does_not_split_the_determinant():
    """``a -> b`` holds exactly when adding ``b`` creates no new groups."""
    profile = _profile(
        row_count=100,
        distinct={"a": 10, "b": 5},
        pair_distinct={frozenset({"a", "b"}): 10},
    )

    assert find_dependencies(profile) == [("a", "b")]


def test_dependency_is_rejected_when_the_pair_splits_the_determinant():
    profile = _profile(
        row_count=100,
        distinct={"a": 10, "b": 5},
        pair_distinct={frozenset({"a", "b"}): 40},
    )

    assert find_dependencies(profile) == []


def test_a_unique_determinant_produces_no_findings():
    """
    A key determines everything, trivially.

    Counting those would swamp every dependency feature with noise that says nothing about
    the normal form.
    """
    profile = _profile(
        row_count=100,
        distinct={"a": 100, "b": 5},
        pair_distinct={frozenset({"a", "b"}): 100},
    )

    assert find_dependencies(profile) == []


# --------------------------------------------------------------------------------------
# Key search (decision E1)
# --------------------------------------------------------------------------------------


def test_key_search_does_not_stop_at_the_first_level():
    """
    The bug this test exists for, measured on the generated data.

    ``note_text`` - a free-text comment - is unique and therefore a perfectly valid
    level-1 UCC. Stopping there hides the real composite key ``{orderkey, linenumber}`` at
    level 2, and with it the partial dependency that defines 2NF: every 1NF table then
    looks like a 2NF one.
    """
    profile = _profile(
        row_count=100,
        distinct={"note_text": 100, "orderkey": 30, "linenumber": 7},
        pair_distinct={
            frozenset({"orderkey", "linenumber"}): 100,
            frozenset({"note_text", "orderkey"}): 100,
            frozenset({"note_text", "linenumber"}): 100,
        },
    )

    uccs, no_key = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert frozenset({"note_text"}) in uccs
    assert frozenset({"orderkey", "linenumber"}) in uccs
    assert not no_key


def test_supersets_of_a_key_are_not_reported():
    """Minimality: ``{note_text, orderkey}`` is unique but contains a smaller UCC."""
    profile = _profile(
        row_count=100,
        distinct={"note_text": 100, "orderkey": 30},
        pair_distinct={frozenset({"note_text", "orderkey"}): 100},
    )

    uccs, _ = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert uccs == [frozenset({"note_text"})]


def test_no_key_found_is_reported_rather_than_hidden():
    """
    The honest "unknown" state from **E1**.

    It has to reach the training data, or the model learns that a key is always available
    and has nothing to fall back on when one is not.
    """
    profile = _profile(
        row_count=100,
        distinct={"a": 5, "b": 4},
        pair_distinct={frozenset({"a", "b"}): 20},
    )

    uccs, no_key = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert uccs == []
    assert no_key


def test_level_three_is_pruned_by_cardinality_before_any_sql():
    """
    ``{a,b,c}`` can only be unique if the distinct counts multiply out to the row count.

    Pure arithmetic on numbers already measured - and with three low-cardinality columns
    there is nothing left to ask the database about.
    """
    profile = _profile(
        row_count=1000,
        distinct={"a": 2, "b": 3, "c": 4},
        pair_distinct={
            frozenset({"a", "b"}): 6,
            frozenset({"a", "c"}): 8,
            frozenset({"b", "c"}): 12,
        },
    )
    conn = FakeConnection()

    uccs, no_key = find_uccs(conn, "t", profile, max_level=3)

    assert uccs == []
    assert no_key
    assert conn.executed == []


# --------------------------------------------------------------------------------------
# 2NF versus 3NF
# --------------------------------------------------------------------------------------


def test_determinant_inside_a_composite_key_is_a_partial_dependency():
    """Half the key determining a non-key attribute - the definition of a 2NF violation."""
    uccs = [frozenset({"orderkey", "linenumber"})]

    classes = classify_dependencies([("orderkey", "orderdate")], uccs, prime=frozenset())

    assert classes.partial == [("orderkey", "orderdate")]
    assert classes.transitive == []


def test_determinant_outside_every_key_is_a_transitive_dependency():
    uccs = [frozenset({"orderkey"})]

    classes = classify_dependencies([("custkey", "mktsegment")], uccs, prime=frozenset())

    assert classes.transitive == [("custkey", "mktsegment")]
    assert classes.partial == []


def test_the_strict_variant_drops_a_dependency_whose_dependent_is_prime():
    """
    The textbook condition, and why it is reported separately.

    It is correct when the candidate keys are known and useless when they are discovered:
    a near-unique column forms a valid UCC with almost every other one, the union of all
    UCCs covers the table, and every attribute comes out prime. Measured on the generated
    set, that drove both strict counts to zero for every single table.
    """
    uccs = [frozenset({"orderkey", "linenumber"})]

    classes = classify_dependencies(
        [("orderkey", "orderdate")],
        uccs,
        prime=frozenset({"orderdate"}),
    )

    assert classes.partial == [("orderkey", "orderdate")]
    assert classes.strict_partial == []


def test_thresholds_are_stated_as_constants():
    """Guards the two numbers the 1NF detector turns on, so a silent edit shows up here."""
    assert LIST_VALUE_MIN_SHARE == 0.8
    # Per token, not per value: a list of 80 part ids is 355 characters long and every
    # piece of it is 4. The old per-value cap of 120 rejected it as prose.
    assert LIST_TOKEN_MAX_MEAN_LENGTH == 20


# --------------------------------------------------------------------------------------
# Ranking the keys, and the two filters that make the ranking meaningful
# --------------------------------------------------------------------------------------


def test_score_prefers_a_structural_key_over_an_accidental_pair():
    """
    The birthday estimate, on the numbers it was derived from.

    ``{orderkey, linenumber}`` has 503 x 7 possible combinations over 2000 rows, so about
    568 collisions were expected and none happened - that uniqueness is structural.
    ``{shipdate, partkey}`` has 1366 x 1266, where uniqueness costs nothing.
    """
    profile = _profile(
        row_count=2000,
        distinct={"orderkey": 503, "linenumber": 7, "shipdate": 1366, "partkey": 1266},
    )

    assert ucc_score(profile, ["orderkey", "linenumber"]) > 500
    assert ucc_score(profile, ["shipdate", "partkey"]) < 2


def test_uccs_come_back_best_first():
    profile = _profile(
        row_count=100,
        distinct={"a": 20, "b": 5, "c": 60, "d": 40},
        pair_distinct={
            frozenset({"a", "b"}): 100,
            frozenset({"c", "d"}): 100,
            frozenset({"a", "c"}): 90,
            frozenset({"a", "d"}): 90,
            frozenset({"b", "c"}): 90,
            frozenset({"b", "d"}): 90,
        },
    )

    uccs, _ = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert uccs[0] == frozenset({"a", "b"})


def test_a_prose_column_is_not_a_key_candidate():
    """
    Measured on 6 of 132 tables: a comment column is unique almost everywhere, which makes
    it a valid UCC and a useless one - and it outranks the real key in every table it
    appears in.

    Ruled out by value statistics rather than by the name veto on purpose: half the
    generated tables carry obfuscated names, where no name-based rule can see anything.
    """
    profile = _profile(
        row_count=100,
        distinct={"note": 100, "orderkey": 30, "linenumber": 7},
        pair_distinct={frozenset({"orderkey", "linenumber"}): 100},
        mean_length=dict.fromkeys(["note", "orderkey", "linenumber"], 48.0),
        mean_separators={"note": 0.16, "orderkey": 0.0, "linenumber": 0.0},
    )
    # Only `note` reads as prose: the other two are numeric, so their measured length is 0.
    profile.mean_length["orderkey"] = 0.0
    profile.mean_length["linenumber"] = 0.0

    uccs, _ = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert frozenset({"note"}) not in uccs
    assert uccs == [frozenset({"orderkey", "linenumber"})]


def test_a_near_unique_column_is_kept_out_of_composite_keys():
    """
    ``{nationkey, supplycost}`` beat the real ``{partkey, suppkey}`` by 40.40 to 40.00.

    ``supplycost`` holds 1980 distinct values over 2000 rows - it does all the separating
    on its own, so pairing it with a 25-value column is unique by arithmetic, not by
    design.
    """
    profile = _profile(
        row_count=2000,
        distinct={"supplycost": 1980, "nationkey": 25, "partkey": 500, "suppkey": 100},
        pair_distinct={
            frozenset({"nationkey", "supplycost"}): 2000,
            frozenset({"partkey", "suppkey"}): 2000,
        },
    )

    uccs, _ = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert uccs == [frozenset({"partkey", "suppkey"})]


def test_a_genuinely_unique_column_is_still_a_key_on_its_own():
    """The near-unique rule applies to composites only - a surrogate key is one column."""
    profile = _profile(
        row_count=100,
        distinct={"id": 100, "name": 40},
        pair_distinct={frozenset({"id", "name"}): 100},
    )

    uccs, _ = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert uccs == [frozenset({"id"})]


# --------------------------------------------------------------------------------------
# Cost budget (a): row sampling
# --------------------------------------------------------------------------------------


def test_a_small_table_is_read_whole():
    source, modulus = _sampled_source("cat.sch.t", ["a", "b"], SAMPLE_ROW_THRESHOLD)

    assert source == "cat.sch.t"
    assert modulus is None


def test_a_large_table_is_sampled_by_a_deterministic_row_hash():
    """
    Not ``TABLESAMPLE``: Trino takes no seed, so it draws different rows every call.

    The profile issues many queries and compares their results against each other, so a
    sample that moves between queries would not be approximate but incoherent.
    """
    total = SAMPLE_ROW_THRESHOLD * 10
    source, modulus = _sampled_source("cat.sch.t", ["a", "b"], total)

    assert modulus == total // SAMPLE_TARGET_ROWS
    assert "TABLESAMPLE" not in source.upper()
    assert "xxhash64" in source
    assert _sampled_source("cat.sch.t", ["a", "b"], total) == (source, modulus)


def _profile_with_recorded_sources(monkeypatch, column_names, total_rows):
    """Profile a fake table, returning the profile and every FROM clause it queried."""
    columns = [ColumnInfo(name=name, data_type="bigint") for name in column_names]
    monkeypatch.setattr(nf_features, "read_columns", lambda *_args, **_kwargs: columns)

    sources: list[str] = []

    def record(_conn, qualified, aggregates, _batch_size, _tolerate=False):
        sources.append(qualified)
        return [SAMPLE_TARGET_ROWS, *([1] * (len(aggregates) - 1))]

    monkeypatch.setattr(nf_features, "_run_aggregates", record)
    profile = profile_table(FakeConnection(scalar_value=total_rows), "cat", "sch", "t")
    return profile, sources


def test_every_count_compared_against_another_shares_one_source(monkeypatch):
    """
    The invariant sampling rests on: counts that get compared come from the same rows.

    ``distinct(a, b) == distinct(a)`` is the test every dependency is derived from. Measure
    the two sides over different row sets and the comparison is meaningless - a pair count
    could even come out below its own single count. So the per-column and the pairwise pass
    share one sampled source, and only the full-table cardinality pass stands apart.
    """
    total = SAMPLE_ROW_THRESHOLD * 10
    profile, sources = _profile_with_recorded_sources(monkeypatch, ("a", "b", "c"), total)

    sampled = [source for source in sources if "xxhash64" in source]
    unsampled = [source for source in sources if "xxhash64" not in source]

    assert len(sampled) > 1, "the per-column and the pair pass both run on the sample"
    assert len(set(sampled)) == 1
    assert unsampled == ["cat.sch.t"], "exactly one pass measures the whole table"
    assert profile.sampled_from == total
    assert profile.total_rows == total
    assert profile.row_count == SAMPLE_TARGET_ROWS


def test_cardinalities_stay_exact_so_the_prune_keeps_its_teeth(monkeypatch):
    """
    Sampling the cardinalities too would gut the one prune that makes level 3 affordable.

    It asks whether the distinct counts multiply out to the row count, so its strength is
    the row count being large. Measured on a fifth of syn_0025, candidates went from 2.331
    to 6.986 and the true key fell from inside the tested 400 to rank 4.679.
    """
    total = SAMPLE_ROW_THRESHOLD * 10
    profile, _ = _profile_with_recorded_sources(monkeypatch, ("a", "b"), total)

    assert profile.distinct_full, "a sampled profile carries full-table cardinalities"
    assert profile.cardinality("a") == profile.distinct_full["a"]

    unsampled = _profile_with_recorded_sources(monkeypatch, ("a", "b"), 100)[0]
    assert unsampled.distinct_full == {}, "without sampling there is nothing to correct"
    assert unsampled.cardinality("a") == unsampled.distinct["a"]


# --------------------------------------------------------------------------------------
# Cost budget (b): bounded search
# --------------------------------------------------------------------------------------


def test_the_pair_search_is_capped_by_column_count(monkeypatch):
    """``C(n,2)`` is the one unbounded cost in the profile - 19.900 aggregates at 200."""
    names = [f"c{index:03d}" for index in range(MAX_PROFILED_COLUMNS + 40)]
    columns = [ColumnInfo(name=name, data_type="bigint") for name in names]
    monkeypatch.setattr(nf_features, "read_columns", lambda *_args, **_kwargs: columns)
    monkeypatch.setattr(
        nf_features,
        "_run_aggregates",
        lambda _conn, _q, aggregates, _b, _t=False: [100, *([1] * (len(aggregates) - 1))],
    )

    profile = profile_table(FakeConnection(scalar_value=100), "cat", "sch", "t")

    measured = set().union(*profile.pair_distinct)
    assert len(measured) == MAX_PROFILED_COLUMNS
    assert profile.pair_coverage < 1.0


def test_a_capped_level_three_search_says_so():
    """
    "Found no key" and "stopped looking" must not arrive at the model as the same number.

    Without ``table_ucc_search_coverage`` both are ``table_has_no_ucc_le3 = 1``, and the
    cap fired on 15 tables of the generated set.
    """
    names = [f"c{index}" for index in range(16)]  # C(16,3) = 560 > the cap
    profile = _profile(
        row_count=1000,
        distinct=dict.fromkeys(names, 500),
        pair_distinct={frozenset(pair): 999 for pair in combinations(names, 2)},
    )
    conn = FakeConnection(rows=[tuple([1] * MAX_LEVEL_3_CANDIDATES)])
    search: dict = {}

    find_uccs(conn, "t", profile, max_level=3, batch_size=MAX_LEVEL_3_CANDIDATES, search=search)

    assert search["candidates"] > MAX_LEVEL_3_CANDIDATES
    assert search["tested"] == MAX_LEVEL_3_CANDIDATES
    assert search["truncated"] is True


def test_a_full_level_three_search_reports_no_truncation():
    names = ["a", "b", "c", "d"]
    profile = _profile(
        row_count=1000,
        distinct=dict.fromkeys(names, 500),
        pair_distinct={frozenset(pair): 999 for pair in combinations(names, 2)},
    )
    search: dict = {}

    find_uccs(FakeConnection(rows=[(1, 1, 1, 1)]), "t", profile, max_level=3, search=search)

    assert search["truncated"] is False
    assert search["tested"] == search["candidates"]


def test_a_constant_column_is_determined_by_everything_and_is_not_a_finding():
    """
    Measured on TPC-H `orders`: `shippriority` is 0 in all 15.000 rows.

    Every other column determines it, which produced seven dependencies - custkey,
    orderstatus, totalprice, orderdate, orderpriority, clerk - and read a 3NF table as 2NF.
    The count is trivial: it says how many columns the table has, not how it is structured.
    """
    profile = _profile(
        row_count=1000,
        distinct={"orderkey": 1000, "custkey": 100, "shippriority": 1},
        pair_distinct={
            frozenset({"orderkey", "custkey"}): 1000,
            frozenset({"orderkey", "shippriority"}): 1000,
            frozenset({"custkey", "shippriority"}): 100,
        },
    )

    assert nf_features.constant_columns(profile) == {"shippriority"}
    assert find_dependencies(profile) == []


def test_a_near_unique_determinant_is_not_a_determinant():
    """
    Measured on TPC-H `customer`: `acctbal` has 1499 distinct values over 1500 rows.

    With that little grouping left there is almost nothing for a dependency to violate, so
    `acctbal -> mktsegment` appears by chance and costs a normal form. The same ratio already
    keeps near-unique columns out of composite keys.
    """
    profile = _profile(
        row_count=1500,
        distinct={"acctbal": 1499, "mktsegment": 5},
        pair_distinct={frozenset({"acctbal", "mktsegment"}): 1499},
    )

    assert find_dependencies(profile) == []
    assert find_near_dependencies(profile) == []


def test_equally_scoring_keys_are_ranked_by_position_not_alphabet():
    """
    On TPC-H `customer`, `custkey` and `address` are both 1500 distinct over 1500 rows.

    Identical `ucc_score`, so the tiebreak decides - and alphabetically `address`, a
    free-text field, became the table's key. Position is a weak signal but not an arbitrary
    one: a key sits at the front of a table by convention.
    """
    profile = _profile(
        row_count=1500,
        distinct={"custkey": 1500, "name": 1500, "address": 1500},
        pair_distinct={},
    )

    uccs, _ = find_uccs(FakeConnection(), "t", profile, max_level=1)

    assert [sorted(ucc) for ucc in uccs] == [["custkey"], ["name"], ["address"]]


def test_a_non_atomic_column_is_not_a_key_candidate():
    """
    A flattened 1:n list is often unique and never a key.

    Left eligible, `lineitem_shipmode_list` outranked the real key in 5 of the 13 tables
    where the ranking failed - and a 0NF table whose "key" is its own list column gets its
    entire dependency classification built on that column.
    """
    names = ["orderkey", "shipmode_list"]
    profile = _profile(
        row_count=100,
        distinct={"orderkey": 100, "shipmode_list": 100},
        pair_distinct={frozenset(names): 100},
        list_like_ratio={"orderkey": 0.0, "shipmode_list": 1.0},
        mean_length={"orderkey": 5.0, "shipmode_list": 18.0},
        mean_separators={"orderkey": 0.0, "shipmode_list": 2.0},
    )

    assert "shipmode_list" in nf_features.non_atomic_columns(profile)
    assert "shipmode_list" not in nf_features.key_eligible_columns(profile)

    uccs, _ = find_uccs(FakeConnection(), "t", profile, max_level=2)
    assert uccs == [frozenset({"orderkey"})]


def test_the_classification_rests_on_the_best_ranked_key_only():
    """
    Decision E2 introduced the ranking; this is the half of the problem it left open.

    With every UCC feeding the position judgement, a table with 17 accidental keys had
    nearly every determinant counted as "part of a key": partial_fd_count 9 against a 1NF
    median of 1, transitive_fd_count 0, prime_ratio 0.69 against 0.3. nf_018 was read as 2NF.
    """
    real_key = frozenset({"a", "b"})
    accidental = frozenset({"c", "d"})
    dependencies = [("c", "e")]  # c sits outside the real key -> transitive, breaks 3NF

    on_ranked = classify_dependencies(dependencies, [real_key], real_key)
    on_flat_set = classify_dependencies(dependencies, [real_key, accidental], real_key | accidental)

    assert on_ranked.transitive == [("c", "e")]
    assert on_ranked.partial == []
    # The old behaviour: the accidental key makes the same dependency look partial instead.
    assert on_flat_set.partial == [("c", "e")]
    assert on_flat_set.transitive == []


# --------------------------------------------------------------------------------------
# The contract between training and serving
# --------------------------------------------------------------------------------------


def test_the_feature_contract_matches_what_is_built(monkeypatch):
    """
    ``FEATURE_COLUMNS`` is what the training X, the request schema and the MLflow signature
    all agree on. If ``build_features`` grows a column and this list does not, training and
    serving drift apart silently - which is finding 1.6 with extra steps.
    """
    columns = [ColumnInfo(name=name, data_type="varchar") for name in ("a", "b")]
    monkeypatch.setattr(nf_features, "read_columns", lambda *_args, **_kwargs: columns)
    monkeypatch.setattr(
        nf_features,
        "_run_aggregates",
        lambda _conn, _q, aggregates, _b, _t=False: [10, *([1] * (len(aggregates) - 1))],
    )

    frame = nf_features.build_features(FakeConnection(scalar_value=10), "cat", "sch", "t")

    assert tuple(frame.columns) == IDENTITY_COLUMNS + FEATURE_COLUMNS


def test_every_float_feature_is_a_feature():
    assert set(FEATURE_COLUMNS) >= FLOAT_FEATURE_COLUMNS


def test_identity_columns_are_not_features():
    assert not set(IDENTITY_COLUMNS) & set(FEATURE_COLUMNS)


# --------------------------------------------------------------------------------------
# Error-tolerant dependencies and keys, the cheap half of (c)
# --------------------------------------------------------------------------------------


def test_an_exact_dependency_has_full_strength():
    profile = _profile(
        row_count=100,
        distinct={"a": 10, "b": 4},
        pair_distinct={frozenset({"a", "b"}): 10},  # adding b splits nothing
    )

    assert dependency_strength(profile, "a", "b") == 1.0
    assert find_near_dependencies(profile) == [], "an exact FD is not a *near* one"


def test_a_dependency_broken_in_a_few_groups_is_near():
    """
    One group of ``a`` split in two: 20 groups where 19 would hold.

    This is what a typo in a dependent column looks like from the outside, and the reason
    (c) exists - on real data a strict test finds nothing at all.
    """
    profile = _profile(
        row_count=1000,
        distinct={"a": 19, "b": 5},
        pair_distinct={frozenset({"a", "b"}): 20},
    )

    near = find_near_dependencies(profile)

    assert dependency_strength(profile, "a", "b") == pytest.approx(0.95)
    assert [(determinant, dependent) for determinant, dependent, _ in near] == [("a", "b")]


def test_a_weak_dependency_is_not_reported():
    profile = _profile(
        row_count=1000,
        distinct={"a": 10, "b": 8},
        pair_distinct={frozenset({"a", "b"}): 60},  # strength 0.167
    )

    assert find_near_dependencies(profile) == []


def test_a_key_determines_everything_and_is_no_near_finding():
    profile = _profile(
        row_count=100,
        distinct={"id": 100, "b": 40},
        pair_distinct={frozenset({"id", "b"}): 100},
    )

    assert find_near_dependencies(profile) == []


def test_a_near_key_is_found_where_no_exact_key_exists():
    """A real primary key with a handful of duplicated rows - the dirty-data case."""
    profile = _profile(
        row_count=1000,
        distinct={"almost": 995, "other": 30},
        pair_distinct={frozenset({"almost", "other"}): 998},
    )

    near = find_near_uccs(profile)
    exact, no_key = find_uccs(FakeConnection(), "t", profile, max_level=2)

    assert frozenset({"almost"}) in near
    assert exact == []
    assert no_key


def test_an_exact_key_is_not_also_a_near_key():
    profile = _profile(
        row_count=100,
        distinct={"id": 100, "b": 40},
        pair_distinct={frozenset({"id", "b"}): 100},
    )

    assert find_near_uccs(profile) == []


def test_a_failed_batch_leaves_the_measurement_unknown_not_zero():
    """
    A timeout must not turn into ``distinct = 0``.

    Zero reads as "constant column", and a constant column is determined by everything -
    the profile would invent a dependency out of a query that never returned.
    """

    class FailingConnection(FakeConnection):
        def execute(self, statement, params=None):  # noqa: ARG002 - matches the real signature
            message = "query exceeded time limit"
            raise OperationalError(str(statement), {}, Exception(message))

    combos = [frozenset({"a", "b"}), frozenset({"a", "c"})]
    counts = count_distinct_combinations(
        FailingConnection(),
        "t",
        combos,
        batch_size=40,
        tolerate_errors=True,
    )

    assert counts == {}

    with pytest.raises(OperationalError):
        count_distinct_combinations(FailingConnection(), "t", combos, batch_size=40)
