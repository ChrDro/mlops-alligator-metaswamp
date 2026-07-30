"""
Fixtures for the Prefect pipeline tests.

No Trino connection is made anywhere in this package. The claim/release helpers in
`prefect/change_events.py` all take an open SQLAlchemy ``Connection``, so a fake that
records the statements it was handed is enough to assert their SQL contract - and it
tests the one property a live database could not show as clearly: that table names
and change types travel as **bound parameters**, never as interpolated SQL.

Import path
-----------
The flow modules import their siblings as top-level modules (``from change_events
import …``), so ``prefect/`` goes on the path. The **repo root must not**: it contains
a directory literally called ``prefect``, which would shadow the installed Prefect
library and break ``from prefect import flow, task`` inside those very modules.
pytest only inserts ``test/`` here (``test/`` has no ``__init__.py`` while this package
does), so that hazard stays theoretical - but it is the reason this file inserts one
specific directory instead of the project root.
"""

import sys
from pathlib import Path


CONFTEST_DIR = Path(__file__).parent
REPO_ROOT = (CONFTEST_DIR / "../..").resolve()
PREFECT_FLOWS_DIR = REPO_ROOT / "prefect"

if str(PREFECT_FLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(PREFECT_FLOWS_DIR))


class FakeResult:
    """Stand-in for a SQLAlchemy ``CursorResult``."""

    def __init__(self, rows=(), scalar_value=None, rowcount: int = 0) -> None:
        self._rows = list(rows)
        self._scalar_value = scalar_value
        self.rowcount = rowcount

    def fetchall(self) -> list:
        return list(self._rows)

    def scalar(self):
        return self._scalar_value


class FakeConnection:
    """
    Records every ``execute`` call as ``(sql, params)`` and returns canned results.

    One canned result serves every statement: the helpers under test each issue at
    most one row-returning query, so per-statement fixtures would add setup without
    adding coverage.
    """

    def __init__(self, rows=(), scalar_value=0, rowcount: int = 0) -> None:
        self.executed: list[tuple[str, dict]] = []
        self._rows = rows
        self._scalar_value = scalar_value
        self._rowcount = rowcount

    def execute(self, statement, params=None) -> FakeResult:
        self.executed.append((str(statement), dict(params or {})))
        return FakeResult(self._rows, self._scalar_value, self._rowcount)

    # --- assertions helpers ---

    def statements(self, containing: str) -> list[str]:
        """Every executed statement containing ``containing`` (case-insensitive)."""
        needle = containing.lower()
        return [sql for sql, _params in self.executed if needle in sql.lower()]

    def params_of(self, containing: str) -> dict:
        """Bound parameters of the first statement containing ``containing``."""
        needle = containing.lower()
        for sql, params in self.executed:
            if needle in sql.lower():
                return params
        message = f"No executed statement contains {containing!r}"
        raise AssertionError(message)

    @property
    def all_sql(self) -> str:
        return "\n".join(sql for sql, _params in self.executed)
