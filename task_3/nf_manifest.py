"""
The manifest for generated normal-form training tables (Phase 2a of TASK_3_PLAN.md).

One row per generated table. It carries everything the feature extractor is **not**
allowed to see: the declared FDs, the resulting label, the generation parameters, and
the review flag from decision **E1**::

    Generator --> Iceberg tables (RAW DATA)  -->  nf_features.py  -->  X
         |                                          (sees ONLY the table)
         +------> Manifest (this module)    -------------------------->  y

Keeping the two apart is what makes "the feature copied the label" structurally
impossible rather than merely forbidden. Nothing under ``prefect/`` should ever import
this module - only the generator and the validation harness (2c) may.

Why the FDs are stored, not just the label
------------------------------------------
2c re-derives the label from the manifest and compares it against FDs discovered on the
materialized table. That only works if the declared FDs round-trip exactly, which is why
``encode_fds``/``decode_fds`` are canonical (sorted) rather than "whatever json.dumps
produced that day".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from nf_labeling import FunctionalDependency, analyze, normalise_fds
from sqlalchemy import text


if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from sqlalchemy.engine import Connection


# Everything Phase 2 generates lives here, well away from the source catalogs the
# pipeline reads in production.
TRAINING_SCHEMA = "iceberg.nf_training"
MANIFEST_TABLE = f"{TRAINING_SCHEMA}.manifest"

MANIFEST_COLUMNS = (
    "database",
    "schema",
    "table_name",
    "target_normal_form",
    "violates_1nf",
    "declared_fds",
    "candidate_keys",
    "generation_params",
    "review_reason",
)


@dataclass(frozen=True)
class ManifestRow:
    """
    One generated table, as the manifest records it.

    Attributes:
        database: Catalog the table was written to ("iceberg").
        schema: Schema inside that catalog ("nf_training").
        table_name: Table name, unique within the schema.
        target_normal_form: The label, 0..3, computed by ``nf_labeling.normal_form``.
        violates_1nf: Whether non-atomic values were injected on purpose. Not derivable
            from the FDs, so it is stored rather than recomputed - 2c needs it to
            reproduce the label.
        declared_fds: Canonical JSON, see ``encode_fds``.
        candidate_keys: Canonical JSON of the keys the FD set implies. Derived, but
            stored anyway: it is exactly what the UCC search from **E1** has to
            reproduce on the raw data, so 2c compares against it.
        generation_params: Free-form JSON (row count, NULL rate, separator, seed, ...).
        review_reason: ``None`` when the table is fine. Non-empty puts it in the manual
            review queue from **E1** - e.g. no UCC found up to level 3.
    """

    database: str
    schema: str
    table_name: str
    target_normal_form: int
    violates_1nf: bool
    declared_fds: str
    candidate_keys: str
    generation_params: str
    review_reason: str | None = None


def encode_fds(fds: Iterable[tuple[Iterable[str], Iterable[str]]]) -> str:
    """
    Serialize FDs canonically: sorted attributes, sorted dependencies.

    Canonical means two runs that declare the same dependencies produce the same string,
    so manifest rows can be diffed and compared without parsing them first.
    """
    encoded = [[sorted(lhs), sorted(rhs)] for lhs, rhs in ((sorted(a), sorted(b)) for a, b in fds)]
    return json.dumps(sorted(encoded), separators=(",", ":"))


def decode_fds(encoded: str) -> list[FunctionalDependency]:
    """Inverse of ``encode_fds`` - the form ``nf_labeling`` takes."""
    return [(frozenset(lhs), frozenset(rhs)) for lhs, rhs in json.loads(encoded)]


def encode_attribute_sets(sets: Iterable[Iterable[str]]) -> str:
    """Canonical JSON for a collection of attribute sets (candidate keys, UCCs)."""
    return json.dumps(sorted(sorted(group) for group in sets), separators=(",", ":"))


def decode_attribute_sets(encoded: str) -> list[frozenset[str]]:
    """Inverse of ``encode_attribute_sets``."""
    return [frozenset(group) for group in json.loads(encoded)]


def build_manifest_row(
    table_name: str,
    attributes: Iterable[str],
    fds: Iterable[tuple[Iterable[str], Iterable[str]]],
    violates_1nf: bool,
    generation_params: dict[str, object],
    database: str = "iceberg",
    schema: str = "nf_training",
    review_reason: str | None = None,
) -> ManifestRow:
    """
    Compute the label and package the row.

    The label is never passed in - it is derived here from the declared FDs, so a
    generator recipe cannot claim a normal form its dependencies do not support.
    """
    normalised = normalise_fds(fds, attributes)
    result = analyze(attributes, normalised, violates_1nf)
    return ManifestRow(
        database=database,
        schema=schema,
        table_name=table_name,
        target_normal_form=result.normal_form,
        violates_1nf=violates_1nf,
        declared_fds=encode_fds(normalised),
        candidate_keys=encode_attribute_sets(result.candidate_keys),
        # The reason behind the label travels with the parameters: cheap to store, and
        # the first thing anyone wants when a label looks wrong.
        generation_params=json.dumps(
            {**generation_params, "label_reason": result.reason},
            separators=(",", ":"),
            sort_keys=True,
            default=str,
        ),
        review_reason=review_reason,
    )


def ensure_manifest_table(conn: Connection) -> None:
    """Create the training schema and the manifest table if they are missing."""
    conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {TRAINING_SCHEMA}"))
    conn.execute(
        text(f"""
            CREATE TABLE IF NOT EXISTS {MANIFEST_TABLE} (
                database VARCHAR,
                schema VARCHAR,
                table_name VARCHAR,
                target_normal_form INTEGER,
                violates_1nf BOOLEAN,
                declared_fds VARCHAR,
                candidate_keys VARCHAR,
                generation_params VARCHAR,
                review_reason VARCHAR
            )
        """),
    )


def insert_manifest_rows(conn: Connection, rows: Sequence[ManifestRow]) -> int:
    """
    Append manifest rows in a single multi-row INSERT.

    Args:
        conn: Open SQLAlchemy connection to Trino.
        rows: Rows to append. An empty sequence is a no-op.

    Returns:
        Number of rows appended.
    """
    if not rows:
        return 0

    ensure_manifest_table(conn)

    value_rows = []
    params: dict[str, object] = {}
    for i, row in enumerate(rows):
        placeholders = []
        for column in MANIFEST_COLUMNS:
            key = f"{column}_{i}"
            params[key] = getattr(row, column)
            placeholders.append(f":{key}")
        value_rows.append(f"({', '.join(placeholders)})")

    columns = ", ".join(MANIFEST_COLUMNS)
    values = ", ".join(value_rows)
    query = f"INSERT INTO {MANIFEST_TABLE} ({columns}) VALUES {values}"  # noqa: S608 - bound parameters
    conn.execute(text(query), params)
    return len(rows)


def delete_manifest_rows(conn: Connection, table_names: Sequence[str]) -> None:
    """
    Drop the manifest rows for these tables.

    Regenerating replaces a table but would *append* its manifest row, so a second run
    would leave two rows per table - with the older one describing data that no longer
    exists. Called by the generator before it inserts.
    """
    if not table_names:
        return

    ensure_manifest_table(conn)
    placeholders = ", ".join(f":name_{i}" for i in range(len(table_names)))
    params = {f"name_{i}": name for i, name in enumerate(table_names)}
    conn.execute(
        text(f"DELETE FROM {MANIFEST_TABLE} WHERE table_name IN ({placeholders})"),  # noqa: S608 - bound parameters
        params,
    )


def load_manifest(conn: Connection, only_review: bool = False) -> list[ManifestRow]:
    """
    Read the manifest back.

    Args:
        conn: Open SQLAlchemy connection to Trino.
        only_review: Return just the manual review queue from **E1** (rows with a
            non-empty ``review_reason``).
    """
    where = "WHERE review_reason IS NOT NULL AND review_reason <> ''" if only_review else ""
    query = f"SELECT {', '.join(MANIFEST_COLUMNS)} FROM {MANIFEST_TABLE} {where}"  # noqa: S608 - constant identifiers
    return [ManifestRow(*row) for row in conn.execute(text(query)).fetchall()]
