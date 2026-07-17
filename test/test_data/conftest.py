
# conftest.py
import pytest

def pytest_addoption(parser):
    """Registers separate command-line options for both test types."""
    parser.addoption(
        "--csv-path-nf",
        action="store",
        default="../../data/nf_test_analyse.csv",  # Default for Normal Form
        help="Path to the CSV file for Normal Form validation"
    )
    parser.addoption(
        "--csv-path-subject",
        action="store",
        default="../../data/raw_metadata.csv",  # Default for Subject Area
        help="Path to the CSV file for Subject Area validation"
    )
    parser.addoption(
        "--csv-path-keys",
        action="store",
        default="../../data/summary_output_task_1_2_training.csv",  # Default for Keys
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
