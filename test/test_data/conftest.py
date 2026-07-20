
# conftest.py
import pytest
from pathlib import Path

CONFTEST_DIR = Path(__file__).parent

DEFAULT_NF_ANALYSE_CSV_PATH = (CONFTEST_DIR / "../../data/nf_test_analyse.csv").resolve()
DEFAULT_TASK_1_2_CSV_PATH = (CONFTEST_DIR / "../../data/summary_output_task_1_2_training.csv").resolve()
DEFAULT_RAW_METADATA_CSV_PATH = (CONFTEST_DIR / "../../data/raw_metadata.csv").resolve()

def pytest_addoption(parser):
    """Registers separate command-line options for both test types."""
    parser.addoption(
        "--csv-path-nf",
        action="store",
        default=str(DEFAULT_NF_ANALYSE_CSV_PATH),  # Default for Normal Form
        help="Path to the CSV file for Normal Form validation"
    )
    parser.addoption(
        "--csv-path-subject",
        action="store",
        default=str(DEFAULT_RAW_METADATA_CSV_PATH),  # Default for Subject Area
        help="Path to the CSV file for Subject Area validation"
    )
    parser.addoption(
        "--csv-path-keys",
        action="store",
        default=str(DEFAULT_TASK_1_2_CSV_PATH),  # Default for Keys
        help="Path to the CSV file for the Keys"
    )

@pytest.fixture(scope="module")
def csv_path_nf(request):
    """Fixture returning the Normal Form CSV path."""
    return request.config.getoption("--csv-path-nf")

@pytest.fixture(scope="module")
def csv_path_subject(request):
    """Fixture returning the Subject Area CSV path."""
    return request.config.getoption("--csv-path-subject")

@pytest.fixture(scope="module")
def csv_path_keys(request):
    """Fixture returning the Keys CSV path."""
    return request.config.getoption("--csv-path-keys")
