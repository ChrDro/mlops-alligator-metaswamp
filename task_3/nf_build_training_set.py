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


def build(out_path: Path, limit: int | None = None) -> pd.DataFrame:
    """Profile every table in the manifest and attach its label."""
    engine = get_trino_engine()
    frames = []

    with engine.connect() as conn:
        rows = load_manifest(conn)
        if limit:
            rows = rows[:limit]
        print(f"{len(rows)} tables in the manifest")

        for index, row in enumerate(rows, start=1):
            features = build_features(conn, row.database, row.schema, row.table_name)
            params = json.loads(row.generation_params)
            features["target_normal_form"] = row.target_normal_form
            # Metadata, not features. The training script drops these; they are here so a
            # split can group on the recipe and an evaluation can pair injections with
            # their controls.
            features["meta_recipe_id"] = params.get("recipe_id")
            features["meta_kind"] = params.get("kind")
            features["meta_pair_id"] = params.get("pair_id")
            features["meta_source_schema"] = params.get("source_schema")
            features["meta_naming"] = params.get("naming")
            frames.append(features)
            print(
                f"  [{index}/{len(rows)}] {row.table_name} "
                f"{row.target_normal_form}NF  {len(features)} columns",
            )

    frame = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_path, index=False)
    print(f"\n{len(frame)} rows -> {out_path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None, help="only the first N tables")
    args = parser.parse_args()
    build(args.out, args.limit)


if __name__ == "__main__":
    main()
