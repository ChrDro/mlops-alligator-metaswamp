"""
Import path and a recording fake connection for the Task 3 modules.

``task_3/`` is a plain directory, not a package (the training script is run as a file),
so the directory itself goes on ``sys.path`` and its modules are imported top-level -
the same arrangement ``test/test_pipelines/conftest.py`` uses for ``prefect/``.

No Trino connection is made anywhere in this package. ``nf_labeling`` is pure Python by
design; ``nf_manifest`` and ``nf_generator`` take an open connection, so a fake that
records the statements it was handed is enough to assert the SQL contract - and it shows
the property a live database would not: that table names travel as **bound parameters**
wherever they can, and are identifier-checked where they cannot.
"""

import sys
from pathlib import Path


CONFTEST_DIR = Path(__file__).parent
REPO_ROOT = (CONFTEST_DIR / "../..").resolve()
TASK_3_DIR = REPO_ROOT / "task_3"

if str(TASK_3_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_3_DIR))


class FakeResult:
    """Stand-in for a SQLAlchemy ``CursorResult``."""

    def __init__(self, rows=()) -> None:
        self._rows = list(rows)

    def fetchall(self):
        return list(self._rows)


class FakeConnection:
    """
    Records every statement and its bound parameters.

    ``rows`` is what any query returns - enough for the two reads in these modules
    (``load_manifest`` and the column check in ``nf_generator.materialise``).
    """

    def __init__(self, rows=()) -> None:
        self.executed: list[tuple[str, dict | None]] = []
        self.rows = list(rows)

    def execute(self, statement, params=None) -> FakeResult:
        self.executed.append((str(statement), params))
        return FakeResult(self.rows)

    def statements(self, containing: str = "") -> list[str]:
        """Every recorded statement, optionally filtered by a substring."""
        return [sql for sql, _ in self.executed if containing in sql]
