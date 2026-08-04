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
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy.exc import DBAPIError


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


class FakeEngine:
    """
    Stand-in for a SQLAlchemy ``Engine`` whose ``begin()`` can fail at commit time.

    ``with_commit_retry`` owns the transaction (a failed iceberg commit leaves the
    old one unusable), so testing it needs an engine rather than a connection. Each
    entry in ``commit_errors`` is raised when the *n*-th transaction exits - which is
    where a real commit conflict surfaces, after the statements already ran.

    ``connections`` holds one :class:`FakeConnection` per attempt, so a test can
    assert the statement really was re-issued on a fresh transaction.
    """

    def __init__(self, commit_errors=(), rows=(), scalar_value=0, rowcount: int = 0) -> None:
        self._commit_errors = list(commit_errors)
        self._rows = rows
        self._scalar_value = scalar_value
        self._rowcount = rowcount
        self.connections: list[FakeConnection] = []

    @contextmanager
    def begin(self):
        connection = FakeConnection(self._rows, self._scalar_value, self._rowcount)
        self.connections.append(connection)
        yield connection
        attempt = len(self.connections) - 1
        if attempt < len(self._commit_errors) and self._commit_errors[attempt] is not None:
            raise self._commit_errors[attempt]

    @property
    def attempts(self) -> int:
        return len(self.connections)


def trino_error(error_name: str, message: str = "boom") -> DBAPIError:
    """
    A SQLAlchemy error wrapping a Trino error, shaped like the real thing.

    The retry decision reads ``orig.error_name``, so the ``orig`` stand-in has to
    carry that attribute - a bare ``Exception`` would silently take the string
    fallback path instead of the one the production code relies on.
    """

    class FakeTrinoError(Exception):
        def __init__(self) -> None:
            super().__init__(f'TrinoExternalError(name={error_name}, message="{message}")')
            self.error_name = error_name

    return DBAPIError("UPDATE …", {}, FakeTrinoError())
