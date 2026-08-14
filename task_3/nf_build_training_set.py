"""
Build the normal-form training set from the generated tables (Phase 2b of TASK_3_PLAN.md).

The only place where features and labels meet. ``nf_features`` produces ``X`` from the
materialized tables alone; the manifest supplies ``y``. They are joined here, on the table
name, and nowhere else - which is what keeps "the feature copied the label" from being
possible rather than merely forbidden.

    python task_3/nf_build_training_set.py --out data/nf_training.csv

The output has one row per column, the same grain the current model is trained on. Whether
that grain is right at all is a Phase 4 question (a table-level model would make
``aggregate_to_table`` unnecessary).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import urllib3


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "prefect"))

from nf_features import build_features  # noqa: E402
from nf_generator import get_trino_engine  # noqa: E402
from nf_manifest import load_manifest  # noqa: E402


urllib3.disable_warnings()

DEFAULT_OUT = REPO_ROOT / "data" / "nf_training.csv"


def discovered_path(out_path: Path) -> Path:
    """Sidecar next to the CSV holding what was discovered per table (for 2c)."""
    return out_path.with_name(out_path.stem + "_discovered.json")


def checkpoint_paths(out_path: Path) -> tuple[Path, Path]:
    """The partial CSV and diagnostics a crashed build leaves behind for ``--resume``."""
    return (
        out_path.with_name(out_path.stem + ".partial.csv"),
        out_path.with_name(out_path.stem + ".partial_discovered.json"),
    )


CHECKPOINT_EVERY = 20


def _flush(frames: list[pd.DataFrame], discovered: dict[str, dict], out_path: Path) -> None:
    """Write everything profiled so far to the checkpoint files."""
    partial_csv, partial_json = checkpoint_paths(out_path)
    partial_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(partial_csv, index=False)
    partial_json.write_text(json.dumps(discovered, indent=1), encoding="utf-8")


def build(
    out_path: Path,
    limit: int | None = None,
    batch_size: int = 20,
    resume: bool = False,
) -> pd.DataFrame:
    """
    Profile every table in the manifest and attach its label.

    ``batch_size`` is 20 rather than the profiler's default 40 because Trino was OOM-killed
    twice at 40 (exit 137, once mid-build at table 75 and once at 129). Each aggregate in a
    batch is a ``count(DISTINCT ...)`` holding its own hash set, so peak memory scales with
    the batch, and the 40-column tables are where it bites. Halving the batch doubles the
    query count and roughly halves the peak - the run gets slower, not smaller.

    Progress is checkpointed every ``CHECKPOINT_EVERY`` tables because a full build is a
    multi-hour Trino conversation and Trino has died in the middle of one four times. The
    fourth cost 290 of 432 profiled tables, purely because the CSV was written after the
    loop. ``resume`` reads the checkpoint back and skips the tables already in it.
    """
    engine = get_trino_engine()
    frames: list[pd.DataFrame] = []
    discovered: dict[str, dict] = {}
    done: set[str] = set()

    partial_csv, partial_json = checkpoint_paths(out_path)
    if resume and partial_csv.exists():
        resumed = pd.read_csv(partial_csv)
        frames.append(resumed)
        done = set(resumed["table_name"])
        if partial_json.exists():
            discovered = json.loads(partial_json.read_text(encoding="utf-8"))
        print(f"resuming from {partial_csv}: {len(done)} tables already profiled")

    with engine.connect() as conn:
        rows = load_manifest(conn)
        if limit:
            rows = rows[:limit]
        print(f"{len(rows)} tables in the manifest")

        for index, row in enumerate(rows, start=1):
            if row.table_name in done:
                continue
            # The discovered keys and dependencies come out of the same pass that builds
            # the features. The validation harness (2c) needs them; recovering them later
            # would mean profiling every table a second time.
            diagnostics: dict = {}
            features = build_features(
                conn,
                row.database,
                row.schema,
                row.table_name,
                diagnostics=diagnostics,
                batch_size=batch_size,
            )
            discovered[row.table_name] = diagnostics
            params = json.loads(row.generation_params)
            features["target_normal_form"] = row.target_normal_form
            # Metadata, not features. The training script drops these; they are here so a
            # split can group on the recipe and an evaluation can pair injections with
            # their controls.
            features["meta_recipe_id"] = params.get("recipe_id")
            features["meta_family"] = params.get("family")
            features["meta_source"] = params.get("source")
            features["meta_kind"] = params.get("kind")
            features["meta_pair_id"] = params.get("pair_id")
            features["meta_source_schema"] = params.get("source_schema")
            features["meta_naming"] = params.get("naming")
            frames.append(features)
            print(
                f"  [{index}/{len(rows)}] {row.table_name} "
                f"{row.target_normal_form}NF  {len(features)} columns",
            )
            if index % CHECKPOINT_EVERY == 0:
                _flush(frames, discovered, out_path)

    frame = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    discovered_path(out_path).write_text(json.dumps(discovered, indent=1), encoding="utf-8")
    # The build finished, so the checkpoint is now a stale copy of the real output. Left in
    # place it would silently short-circuit the next full build via --resume.
    for stale in checkpoint_paths(out_path):
        stale.unlink(missing_ok=True)
    print(f"\n{len(frame)} rows -> {out_path}")
    print(f"{len(discovered)} table diagnostics -> {discovered_path(out_path)}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tables")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="aggregates per profiling query; lower it if Trino runs out of memory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from the checkpoint a crashed build left behind, skipping its tables",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="skip the validation harness (Phase 2c) - it normally runs with every build",
    )
    parser.add_argument(
        "--write-review",
        action="store_true",
        help="file the harness offenders in the manifest's review_reason (the E1 queue)",
    )
    args = parser.parse_args()
    build(args.out, args.limit, args.batch_size, resume=args.resume)

    if args.no_validate:
        print("\nvalidation skipped (--no-validate)")
        return

    # Deferred: nf_validate imports discovered_path from here, so a top-level import in
    # both directions would be a cycle. The builder owns the output paths, the harness
    # owns the checks; this is the seam.
    from nf_validate import validate  # noqa: PLC0415 - see comment

    print("\n" + "=" * 70)
    results = validate(training_set=args.out, write_review=args.write_review)
    if any(not result.passed for result in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
