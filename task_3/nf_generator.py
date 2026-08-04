"""
Generator skeleton for normal-form training tables (Phase 2a of TASK_3_PLAN.md).

What this file is
-----------------
The plumbing, not the recipes. It materializes a ``TableSpec`` as a real Iceberg table,
derives the label from the declared FDs via ``nf_labeling``, and writes the manifest row.
``RECIPES`` is deliberately empty: filling it is Phase **2b**, where the TPC-H joins and
the 1NF injections are worked out.

Why raw tables and not feature rows
-----------------------------------
The current dataset persisted only aggregates (finding 1.7). That is why the three open
work items - measured ``table_ratio_1nf_violations``, column-pair features, atomicity
features - all hit the same wall: the values they need are gone. Everything here writes
**rows**, so a feature added later can simply be recomputed.

The rule this file exists to enforce
------------------------------------
The label is derived here, from the FDs, and lands in the manifest - never in a column
of the generated table. The feature extractor gets the materialized table and nothing
else, the same view it has in production.

Run it
------
``python task_3/nf_generator.py --dry-run`` computes and prints the labels without
touching Trino. That is the useful mode until 2b fills in the recipes.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from nf_labeling import analyze, normalise_fds
from nf_manifest import (
    TRAINING_SCHEMA,
    ManifestRow,
    build_manifest_row,
    delete_manifest_rows,
    insert_manifest_rows,
)
from sqlalchemy import create_engine, text


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.engine import Connection, Engine


load_dotenv()

TRINO_IP_ADDRESS = str(os.environ.get("TRINO_IP_ADDRESS"))
TRINO_USERNAME = str(os.environ.get("TRINO_USERNAME"))
TRINO_PASSWORD = str(os.environ.get("TRINO_PASSWORD"))

# Table names are interpolated into DDL (they cannot be bound parameters), so they are
# checked against this instead of trusted.
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class TableSpec:
    """
    One table to generate.

    Attributes:
        table_name: Target name inside ``nf_training``. Must be a plain identifier.
        source_sql: A ``SELECT`` producing the raw rows. Its output columns must match
            ``attributes`` exactly - that is verified after materialization.
        attributes: Every column of the generated table.
        declared_fds: The dependencies the data is built to satisfy. The label comes
            from these; 2c checks them back against the materialized rows.
        violates_1nf: True when non-atomic values were injected on purpose. Cannot be
            derived from the FDs, so the recipe has to state it.
        generation_params: Anything worth recording (row count, separator, seed, source
            query). Free-form, ends up as JSON in the manifest.
        review_reason: Pre-set only when the recipe already knows the table needs a look
            (see **E1**); the UCC search fills this in later for the rest.
    """

    table_name: str
    source_sql: str
    attributes: tuple[str, ...]
    declared_fds: tuple[tuple[frozenset[str], frozenset[str]], ...] = ()
    violates_1nf: bool = False
    generation_params: dict[str, object] = field(default_factory=dict)
    review_reason: str | None = None


def load_specs() -> list[TableSpec]:
    """
    The recipes from Phase 2b.

    Imported here rather than at module level because ``nf_recipes`` needs ``TableSpec``
    from this module - a top-level import in both directions would be a cycle. The
    generator owns the plumbing, the recipes own the shapes; this is the seam.
    """
    from nf_recipes import build_specs  # noqa: PLC0415 - see docstring

    return build_specs()


def get_trino_engine() -> Engine:
    """Same connection settings the Prefect flows use."""
    return create_engine(
        f"trino://{TRINO_USERNAME}:{TRINO_PASSWORD}@{TRINO_IP_ADDRESS}:8443/iceberg",
        connect_args={
            "http_scheme": "https",
            "verify": False,
            "session_properties": {"distinct_aggregations_strategy": "single_step"},
        },
    )


def _qualified(table_name: str) -> str:
    if not IDENTIFIER.match(table_name):
        message = f"{table_name!r} is not a plain SQL identifier"
        raise ValueError(message)
    return f"{TRAINING_SCHEMA}.{table_name}"


def materialise(conn: Connection, spec: TableSpec) -> None:
    """
    Write the raw rows as an Iceberg table, replacing any earlier version.

    Regenerating has to be repeatable, so the table is dropped first rather than
    appended to - a half-overwritten table would silently break the FDs the label
    claims and there is nothing downstream that could notice.
    """
    target = _qualified(spec.table_name)
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {TRAINING_SCHEMA}"))
    conn.execute(text(f"DROP TABLE IF EXISTS {target}"))
    conn.execute(text(f"CREATE TABLE {target} AS {spec.source_sql}"))
    _verify_columns(conn, spec)


def _verify_columns(conn: Connection, spec: TableSpec) -> None:
    """
    The materialized columns must be exactly the declared attributes.

    Without this check a renamed or extra column in ``source_sql`` produces FDs that
    reference something the table does not have - and since a non-firing FD only makes
    the table look *better* normalized, the label would be wrong in the quiet direction.
    """
    rows = conn.execute(
        text("""
            SELECT column_name
            FROM iceberg.information_schema.columns
            WHERE table_schema = :schema AND table_name = :table_name
        """),
        {"schema": TRAINING_SCHEMA.split(".")[-1], "table_name": spec.table_name},
    ).fetchall()

    actual = {row[0] for row in rows}
    declared = set(spec.attributes)
    if actual != declared:
        message = (
            f"{spec.table_name}: materialized columns {sorted(actual)} do not match the "
            f"declared attributes {sorted(declared)}"
        )
        raise ValueError(message)


def build_row(spec: TableSpec) -> ManifestRow:
    """Derive the label from the declared FDs and package the manifest row."""
    return build_manifest_row(
        table_name=spec.table_name,
        attributes=spec.attributes,
        fds=spec.declared_fds,
        violates_1nf=spec.violates_1nf,
        generation_params=spec.generation_params,
        review_reason=spec.review_reason,
    )


def generate(conn: Connection, specs: Sequence[TableSpec]) -> list[ManifestRow]:
    """
    Materialize every spec and record it in the manifest.

    Labels are computed **before** anything is written: a broken FD declaration should
    fail on the first spec, not after half the schema has been rebuilt.

    Returns:
        The manifest rows that were written.
    """
    rows = [build_row(spec) for spec in specs]

    for spec in specs:
        materialise(conn, spec)
        print(f"materialised {_qualified(spec.table_name)}")

    # Regenerating replaces the table but would append its manifest row; without this a
    # second run leaves a stale row describing data that no longer exists.
    delete_manifest_rows(conn, [spec.table_name for spec in specs])
    written = insert_manifest_rows(conn, rows)
    print(f"wrote {written} manifest rows")
    return rows


def label_of(spec: TableSpec) -> int:
    """The label this spec would get, without touching the database."""
    return analyze(
        spec.attributes,
        normalise_fds(spec.declared_fds, spec.attributes),
        spec.violates_1nf,
    ).normal_form


def dry_run(specs: Sequence[TableSpec], verbose: bool = False) -> None:
    """Print the label each spec would get, and the class distribution."""
    if not specs:
        print("No recipes defined yet - see nf_recipes (Phase 2b).")
        return

    distribution: dict[int, int] = {}
    for spec in specs:
        result = analyze(
            spec.attributes,
            normalise_fds(spec.declared_fds, spec.attributes),
            spec.violates_1nf,
        )
        distribution[result.normal_form] = distribution.get(result.normal_form, 0) + 1
        if verbose:
            keys = " | ".join("{" + ", ".join(sorted(key)) + "}" for key in result.candidate_keys)
            recipe = spec.generation_params.get("recipe_id", "?")
            print(f"{spec.table_name:<24} {result.normal_form}NF  {recipe:<32} keys: {keys}")
            print(f"{'':<24} {result.reason}")

    total = len(specs)
    print(f"\n{total} tables")
    for form in sorted(distribution):
        count = distribution[form]
        print(f"  {form}NF: {count:>4}  ({count / total:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and print labels without writing anything",
    )
    parser.add_argument("--verbose", action="store_true", help="one line per table")
    parser.add_argument(
        "--only",
        default=None,
        help="restrict to table names containing this substring",
    )
    args = parser.parse_args()

    specs = load_specs()
    if args.only:
        specs = [spec for spec in specs if args.only in spec.table_name]

    if args.dry_run:
        dry_run(specs, verbose=args.verbose)
        return

    if not specs:
        print("Nothing to generate.")
        return

    engine = get_trino_engine()
    with engine.connect() as conn:
        generate(conn, specs)


if __name__ == "__main__":
    main()
