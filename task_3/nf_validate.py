"""
Validation harness for the generated training set (Phase 2c of TASK_3_PLAN.md).

Runs after every generation and answers one question in six parts: is this dataset
*worth training on*, or does it contain the same kind of shortcut the last one did?

    python task_3/nf_validate.py                 # report
    python task_3/nf_validate.py --write-review  # and file the offenders under E1

The checks, and what each one would have caught in the old data:

| check                | would have caught                                                |
| :------------------- | :--------------------------------------------------------------- |
| label verification   | a declared FD the data does not satisfy - the label is then wrong |
| signature test       | `table_avg_unique_ratio` separating 2NF from 3NF (finding 1.3)   |
| decoy test           | a 1NF detector firing on comment columns                          |
| label noise          | the six `tbl_dirty_1nf_*` tables with no violating column         |
| key search           | tables where no key was found at all (decision **E1**)            |
| key ranking          | the real key drowning among 18-34 accidental UCCs (**E2**)        |

Three of them need the manifest or a connection, three the built feature table.
Failures do not delete anything: offenders are written into the manifest's
``review_reason``, which is the manual review queue **E1** created the column for.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import urllib3


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "prefect"))

from nf_build_training_set import discovered_path  # noqa: E402
from nf_features import count_distinct_combinations  # noqa: E402
from nf_generator import get_trino_engine  # noqa: E402
from nf_labeling import normal_form, normalise_fds  # noqa: E402
from nf_manifest import (  # noqa: E402
    decode_attribute_sets,
    decode_fds,
    load_manifest,
    update_review_reasons,
)


if TYPE_CHECKING:
    from collections.abc import Sequence

    from nf_manifest import ManifestRow
    from sqlalchemy.engine import Connection


urllib3.disable_warnings()

DEFAULT_TRAINING_SET = REPO_ROOT / "data" / "nf_training.csv"

# A single feature reaching this weighted F1 on its own is a generator signature, not a
# property of normal forms. The old set failed this badly: `table_ratio_1nf_violations`
# alone determined class 0 with 100% agreement.
SIGNATURE_F1_LIMIT = 0.6

# Below this, the 1NF detector misses too many injected violations to be worth trusting.
# Not 1.0: an obfuscated repeating group is genuinely invisible - a repeating group lives
# in the column names, and obfuscated names do not have any.
DETECTOR_MIN_RECALL = 0.9

# No tolerance here at all. A false positive flips a whole table to 0NF, and the fragility
# simulation in finding 1.5 put the cost at F1 0.9939 -> 0.3143.
DETECTOR_MIN_PRECISION = 1.0

# The real key must be *among* the discovered UCCs for every table - measured at 132/132,
# so anything less is a regression in the search or an E2 filter that went too far.
KEY_RANKING_MIN_FOUND = 1.0

# Ranked first: measured at 128/132. The bar sits below that because the two E2 thresholds
# are calibrated on 132 tables and will have to move at 2d scale - this check is what makes
# that move visible instead of silent.
KEY_RANKING_MIN_FIRST = 0.9


@dataclass
class CheckResult:
    """One check: did it pass, what did it measure, and which tables are at fault."""

    name: str
    passed: bool
    detail: str
    offenders: dict[str, str] = field(default_factory=dict)

    def render(self) -> str:
        mark = "PASS" if self.passed else "FAIL"
        lines = [f"[{mark}] {self.name}: {self.detail}"]
        for table, reason in sorted(self.offenders.items()):
            lines.append(f"         {table}: {reason}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# 1. Label verification - needs the data
# --------------------------------------------------------------------------------------


def verify_labels(
    conn: Connection,
    rows: Sequence[ManifestRow],
    sample: int | None = None,
) -> CheckResult:
    """
    Do the declared dependencies actually hold, and would the discovered ones change the
    label?

    Two directions, and they fail for different reasons:

    * **A declared FD that does not hold** is a generator bug. The label was computed from
      dependencies the data does not satisfy, so it is simply wrong. Hard failure.
    * **Discovered FDs beyond the declared ones** are normal - a finite table always
      satisfies accidental dependencies. Reporting their mere existence would cry wolf on
      every run. What matters is whether adding them *changes the label*, and that is what
      is checked.

    Args:
        conn: Open SQLAlchemy connection to Trino.
        rows: Manifest rows to verify.
        sample: Verify only the first N tables. The full run is one query per table.
    """
    checked = rows[:sample] if sample else rows
    offenders: dict[str, str] = {}

    for row in checked:
        declared = decode_fds(row.declared_fds)
        if not declared:
            continue

        qualified = f"{row.database}.{row.schema}.{row.table_name}"
        combos = {lhs for lhs, _ in declared} | {lhs | rhs for lhs, rhs in declared}
        counts = count_distinct_combinations(conn, qualified, sorted(combos, key=sorted))

        broken = [
            (sorted(lhs), sorted(rhs))
            for lhs, rhs in declared
            if counts.get(lhs | rhs, -1) != counts.get(lhs, -2)
        ]
        if broken:
            lhs, rhs = broken[0]
            offenders[row.table_name] = (
                f"declared FD {{{', '.join(lhs)}}} -> {{{', '.join(rhs)}}} does not hold "
                f"in the data ({len(broken)} of {len(declared)} broken)"
            )

    passed = not offenders
    detail = (
        f"{len(checked) - len(offenders)}/{len(checked)} tables satisfy every declared dependency"
    )
    return CheckResult("label verification", passed, detail, offenders)


def verify_label_against_discovered(
    rows: Sequence[ManifestRow],
    discovered: dict[str, dict],
) -> CheckResult:
    """
    Recompute each label with the *discovered* dependencies added, and compare.

    The completeness half of the check above, made actionable. Extra dependencies always
    exist - a finite table satisfies accidental ones no schema ever declared - so merely
    reporting that they exist would cry wolf on every run. What matters is whether they are
    strong enough to move the label.

    A moved label means one of two things, and both belong in the review queue: the sample
    is too small and accidental dependencies dominate, or the recipe declared less than it
    actually built.
    """
    offenders: dict[str, str] = {}
    checked = 0

    for row in rows:
        diagnostics = discovered.get(row.table_name)
        if not diagnostics:
            continue
        checked += 1

        declared = decode_fds(row.declared_fds)
        extra = [
            (frozenset({determinant}), frozenset({dependent}))
            for determinant, dependent in diagnostics["dependencies"]
        ]
        # Pair-determinant findings (2026-08-07) go through the same question: not "do
        # extras exist" but "would they move the label".
        extra += [
            (frozenset(determinant), frozenset({dependent}))
            for determinant, dependent in diagnostics.get("pair_dependencies", [])
        ]
        combined = normalise_fds([*declared, *extra], diagnostics["columns"])
        recomputed = normal_form(diagnostics["columns"], combined, row.violates_1nf)

        if recomputed != row.target_normal_form:
            offenders[row.table_name] = (
                f"label {row.target_normal_form}NF, but {recomputed}NF once the "
                f"{len(extra)} discovered dependencies are added"
            )

    return CheckResult(
        "label vs discovered dependencies",
        passed=not offenders,
        detail=f"{checked - len(offenders)}/{checked} labels unchanged by what was discovered",
        offenders=offenders,
    )


# --------------------------------------------------------------------------------------
# 2. Signature test - runs on the feature table
# --------------------------------------------------------------------------------------


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Everything that is a feature: not identity, not metadata, not the label."""
    excluded = {
        "database",
        "schema",
        "table_name",
        "column_name",
        "column_type",
        "target_normal_form",
    }
    return [
        column
        for column in frame.columns
        if column not in excluded and not column.startswith("meta_")
    ]


def detect_signatures(frame: pd.DataFrame, limit: float = SIGNATURE_F1_LIMIT) -> CheckResult:
    """
    A decision stump on each feature alone. Anything above ``limit`` is a shortcut.

    The point is not that one strong feature is bad - it is that a *single* feature
    reaching two thirds of the way to a perfect score means the generator wrote the answer
    into the data. The old set failed this test spectacularly, which is what finding 1.3
    is about.

    Grouped by recipe, so a stump cannot win by memorising a table shape it has already
    seen at a different scale factor.
    """
    from sklearn.metrics import f1_score  # noqa: PLC0415 - heavy import, only needed here
    from sklearn.model_selection import GroupKFold  # noqa: PLC0415
    from sklearn.tree import DecisionTreeClassifier  # noqa: PLC0415

    features = feature_columns(frame)
    y = frame["target_normal_form"]
    groups = frame["meta_recipe_id"].fillna("?")

    scores: dict[str, float] = {}
    for feature in features:
        fold_scores = []
        for train, test in GroupKFold(n_splits=5).split(frame, y, groups):
            stump = DecisionTreeClassifier(max_depth=2, random_state=42)
            stump.fit(frame.iloc[train][[feature]], y.iloc[train])
            fold_scores.append(
                f1_score(
                    y.iloc[test],
                    stump.predict(frame.iloc[test][[feature]]),
                    average="weighted",
                ),
            )
        scores[feature] = float(np.mean(fold_scores))

    over = {
        name: f"{score:.4f} weighted F1 on its own"
        for name, score in scores.items()
        if score > limit
    }
    best = max(scores.items(), key=lambda item: item[1])
    return CheckResult(
        "signature test",
        passed=not over,
        detail=f"strongest single feature: {best[0]} at {best[1]:.4f} (limit {limit})",
        offenders=over,
    )


# --------------------------------------------------------------------------------------
# 3. Decoy test - the 1NF detector, reported on its own
# --------------------------------------------------------------------------------------


def detector_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Per table: did the atomicity detector fire, and what is the truth."""
    return frame.groupby("table_name").agg(
        truth=("target_normal_form", "first"),
        list_columns=("table_ratio_list_like_columns", "first"),
        repeating=("table_repeating_group_ratio", "first"),
        recipe=("meta_recipe_id", "first"),
        naming=("meta_naming", "first"),
    )


def decoy_report(
    frame: pd.DataFrame,
    min_recall: float = DETECTOR_MIN_RECALL,
    min_precision: float = DETECTOR_MIN_PRECISION,
) -> CheckResult:
    """
    Precision and recall of the 1NF detector, **separately** from the four-class score.

    Smeared into a weighted F1 the detector's behaviour is invisible, and it is the one
    component where the two error directions cost wildly different amounts: a miss loses
    one table, a false positive flips a whole table to 0NF (finding 1.5 measured that at
    F1 0.9939 -> 0.3143). Precision therefore has no tolerance and recall does.
    """
    tables = detector_flags(frame)
    fired = (tables.list_columns > 0) | (tables.repeating > 0)
    is_zero_nf = tables.truth == 0

    true_positive = int((fired & is_zero_nf).sum())
    false_positive = int((fired & ~is_zero_nf).sum())
    false_negative = int((~fired & is_zero_nf).sum())

    precision = true_positive / (true_positive + false_positive) if fired.any() else 1.0
    recall = true_positive / (true_positive + false_negative) if is_zero_nf.any() else 1.0

    offenders = {
        table: f"{'false positive' if row.truth != 0 else 'missed 0NF'} "
        f"(recipe {row.recipe}, naming {row.naming})"
        for table, row in tables[(fired & ~is_zero_nf) | (~fired & is_zero_nf)].iterrows()
    }

    passed = precision >= min_precision and recall >= min_recall
    return CheckResult(
        "decoy test",
        passed,
        f"precision {precision:.4f} (>= {min_precision}), recall {recall:.4f} "
        f"(>= {min_recall}) over {len(tables)} tables",
        offenders,
    )


# --------------------------------------------------------------------------------------
# 4. Label noise
# --------------------------------------------------------------------------------------


def check_label_noise(frame: pd.DataFrame) -> CheckResult:
    """
    No table may be labelled 0NF without something in it that a feature can see.

    This is the exact defect of the six ``tbl_dirty_1nf_*`` tables in the current set: 0NF
    by label, with not a single flagged column. A model cannot learn that, it can only
    memorise it - which is what it did.

    Known and accepted: obfuscated repeating groups. A repeating group exists only in the
    column names, so once the names are gone, so is the evidence. Those go to review rather
    than being called a bug.
    """
    tables = detector_flags(frame)
    fired = (tables.list_columns > 0) | (tables.repeating > 0)
    silent = tables[(tables.truth == 0) & ~fired]

    explained = silent[silent.naming == "obfuscated"]
    unexplained = silent[silent.naming != "obfuscated"]

    offenders = {
        table: (
            "0NF with no visible violation - obfuscated names, so a repeating group cannot be seen"
            if table in explained.index
            else "0NF with no visible violation and no explanation"
        )
        for table in silent.index
    }
    return CheckResult(
        "label noise",
        passed=unexplained.empty,
        detail=(
            f"{len(silent)} of {int((tables.truth == 0).sum())} 0NF tables carry no visible "
            f"violation ({len(explained)} explained by obfuscated naming, "
            f"{len(unexplained)} unexplained)"
        ),
        offenders=offenders,
    )


# --------------------------------------------------------------------------------------
# 5. Key search (decision E1)
# --------------------------------------------------------------------------------------


def check_key_search(frame: pd.DataFrame) -> CheckResult:
    """
    Tables where the UCC search came back empty up to level 3.

    **E1** settled this: report and carry on, then look at them by hand. The check exists
    so "carry on" does not quietly become "ignore" - and so the ``unknown`` state reaches
    the training data instead of being papered over.
    """
    tables = frame.groupby("table_name").agg(
        no_key=("table_has_no_ucc_le3", "first"),
        columns=("table_column_count", "first"),
        rows=("table_row_count", "first"),
    )
    missing = tables[tables.no_key == 1]

    offenders = {
        table: f"no UCC up to level 3 ({int(row.columns)} columns, {int(row.rows)} rows)"
        for table, row in missing.iterrows()
    }
    return CheckResult(
        "key search",
        passed=True,  # E1: reporting is the outcome, not failing
        detail=f"{len(missing)}/{len(tables)} tables have no key up to level 3",
        offenders=offenders,
    )


# --------------------------------------------------------------------------------------
# 6. Key ranking (decision E2)
# --------------------------------------------------------------------------------------


def check_key_ranking(
    rows: Sequence[ManifestRow],
    discovered: dict[str, dict],
    min_found: float = KEY_RANKING_MIN_FOUND,
    min_first: float = KEY_RANKING_MIN_FIRST,
) -> CheckResult:
    """
    Is the real key among the discovered UCCs, and does the ranking put it on top?

    The check **E2** exists for. It is the only one here with access to a ground truth the
    extractor never sees: ``candidate_keys`` in the manifest, derived from the declared FD
    set. Everything else about the key search is a judgement call; this is a measurement.

    Two numbers, and they fail for different reasons:

    * **found** - the true key is not among the discovered UCCs at all. Either the search
      is broken or one of the E2 filters threw away a column the key needs. Measured at
      132/132, so the bar is 1.0: any drop is a regression, not noise.
    * **first** - it was found but something outranked it. Measured at 128/132; the four
      misses are genuine near-ties. The bar sits below that on purpose, because the
      thresholds behind the ranking (20 and 0.9) are calibrated on 132 tables and will
      move at 2d scale.

    Relies on ``uccs`` arriving best-first, which is what ``nf_features.find_uccs``
    guarantees since E2. That coupling is the reason the score is not recomputed here -
    recomputing it would test this module against itself instead of against the ranking
    the features were actually built from.
    """
    found = 0
    first = 0
    checked = 0
    offenders: dict[str, str] = {}

    for row in rows:
        diagnostics = discovered.get(row.table_name)
        if not diagnostics or not diagnostics.get("uccs"):
            continue

        truth = {frozenset(key) for key in decode_attribute_sets(row.candidate_keys)}
        if not truth:
            continue
        checked += 1

        ranked = [frozenset(ucc) for ucc in diagnostics["uccs"]]
        hit = [index for index, ucc in enumerate(ranked) if ucc in truth]
        if not hit:
            offenders[row.table_name] = (
                f"true key {[sorted(k) for k in truth]} is not among the "
                f"{len(ranked)} discovered UCCs"
            )
            continue

        found += 1
        if hit[0] == 0:
            first += 1
        else:
            offenders[row.table_name] = (
                f"true key ranked {hit[0] + 1} of {len(ranked)}; outranked by {sorted(ranked[0])}"
            )

    if not checked:
        return CheckResult("key ranking", True, "no diagnostics to check")

    found_share = found / checked
    first_share = first / checked
    return CheckResult(
        "key ranking",
        passed=found_share >= min_found and first_share >= min_first,
        detail=(
            f"true key found in {found}/{checked} ({found_share:.4f} >= {min_found}), "
            f"ranked first in {first}/{checked} ({first_share:.4f} >= {min_first})"
        ),
        offenders=offenders,
    )


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


def run_all(
    frame: pd.DataFrame,
    conn: Connection | None = None,
    discovered: dict[str, dict] | None = None,
    sample: int | None = None,
) -> list[CheckResult]:
    """
    Every check, in the order a failure should be read.

    Label verification comes first: if the declared dependencies do not hold, nothing
    downstream means anything, because the labels themselves are wrong.
    """
    results = []
    if conn is not None:
        rows = load_manifest(conn)
        results.append(verify_labels(conn, rows, sample=sample))
        if discovered:
            results.append(verify_label_against_discovered(rows, discovered))
            results.append(check_key_ranking(rows, discovered))
    results += [
        detect_signatures(frame),
        decoy_report(frame),
        check_label_noise(frame),
        check_key_search(frame),
    ]
    return results


def collect_review_reasons(results: list[CheckResult]) -> dict[str, str]:
    """Fold every offender into one review note per table, for the **E1** queue."""
    reasons: dict[str, list[str]] = {}
    for result in results:
        for table, reason in result.offenders.items():
            reasons.setdefault(table, []).append(f"{result.name}: {reason}")
    return {table: " | ".join(notes) for table, notes in reasons.items()}


def validate(
    training_set: Path = DEFAULT_TRAINING_SET,
    write_review: bool = False,
    offline: bool = False,
    sample: int | None = None,
) -> list[CheckResult]:
    """
    Run every check and print the report.

    The entry point for both the CLI and the training-set builder, which calls it after
    every build - a harness that has to be remembered is a harness that does not run.

    Args:
        training_set: The built feature table. Its diagnostics sidecar is picked up next
            to it, if the build wrote one.
        write_review: File the offenders in the manifest's ``review_reason`` (**E1**).
        offline: Skip the two checks that need a Trino connection.
        sample: Verify the declared FDs of only the first N tables.

    Returns:
        One result per check, in the order a failure should be read.
    """
    frame = pd.read_csv(training_set)
    sidecar = discovered_path(training_set)
    discovered = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else {}
    print(f"{len(frame)} rows, {frame.table_name.nunique()} tables")
    if not discovered:
        print(f"no diagnostics at {sidecar} - rebuild the training set to enable one check")
    print()

    if offline:
        results = run_all(frame)
    else:
        engine = get_trino_engine()
        with engine.connect() as conn:
            results = run_all(frame, conn, discovered=discovered, sample=sample)
            if write_review:
                reasons = collect_review_reasons(results)
                update_review_reasons(conn, reasons, all_tables=list(frame.table_name.unique()))
                print(f"filed {len(reasons)} tables for review\n")

    for result in results:
        print(result.render())

    failed = [result.name for result in results if not result.passed]
    print()
    print(f"FAILED: {', '.join(failed)}" if failed else "all checks passed")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-set", type=Path, default=DEFAULT_TRAINING_SET)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="verify the declared FDs of only the first N tables",
    )
    parser.add_argument(
        "--write-review",
        action="store_true",
        help="write the offenders into the manifest's review_reason (the E1 queue)",
    )
    parser.add_argument("--offline", action="store_true", help="skip the checks that need Trino")
    args = parser.parse_args()

    results = validate(
        training_set=args.training_set,
        write_review=args.write_review,
        offline=args.offline,
        sample=args.sample,
    )
    if any(not result.passed for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
