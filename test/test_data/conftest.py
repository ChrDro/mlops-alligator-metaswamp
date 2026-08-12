# conftest.py
from pathlib import Path

import pytest


CONFTEST_DIR = Path(__file__).parent

DEFAULT_TASK_1_2_CSV_PATH = (
    CONFTEST_DIR / "../../data/summary_output_task_1_2_training.csv"
).resolve()
DEFAULT_RAW_METADATA_CSV_PATH = (CONFTEST_DIR / "../../data/raw_metadata.csv").resolve()


def pytest_addoption(parser):
    """Registers separate command-line options for both test types."""
    parser.addoption(
        "--csv-path-subject",
        action="store",
        default=str(DEFAULT_RAW_METADATA_CSV_PATH),  # Default for Subject Area
        help="Path to the CSV file for Subject Area validation",
    )
    parser.addoption(
        "--csv-path-keys",
        action="store",
        default=str(DEFAULT_TASK_1_2_CSV_PATH),  # Default for Keys
        help="Path to the CSV file for the Keys",
    )


@pytest.fixture(scope="module")
def csv_path_subject(request: pytest.FixtureRequest) -> Path:
    """Fixture returning the Subject Area CSV path."""
    return Path(request.config.getoption("--csv-path-subject"))


@pytest.fixture(scope="module")
def csv_path_keys(request: pytest.FixtureRequest) -> Path:
    """Fixture returning the Keys CSV path."""
    return Path(request.config.getoption("--csv-path-keys"))
