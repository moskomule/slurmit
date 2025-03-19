import os
import pathlib
import tempfile
from unittest import mock

import cloudpickle
import pytest

from slurmit import JobStatus, Job, SlurmExecutor


@pytest.fixture
def mock_slurm_commands():
    """Mock SLURM command-line tools for testing."""
    with mock.patch('subprocess.run') as mock_run:
        # Configure mock to return appropriate responses for different commands
        def side_effect(*args, **kwargs):
            command = args[0][0] if args and isinstance(args[0], list) else None

            # Mock sbatch version check
            if command == 'sbatch' and '--version' in args[0]:
                result = mock.MagicMock()
                result.stdout = "slurm 23.02.0"
                return result

            # Mock job submission
            elif command == 'sbatch' and '--version' not in args[0]:
                result = mock.MagicMock()
                result.stdout = "Submitted batch job 12345"
                return result

            # Mock job status check
            elif command == 'sacct':
                result = mock.MagicMock()
                result.stdout = "COMPLETED\n"
                return result

            return mock.MagicMock()

        mock_run.side_effect = side_effect
        yield mock_run


@pytest.fixture
def temp_executor(mock_slurm_commands):
    """Create a temporary SlurmExecutor for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a basic template file
        template_path = os.path.join(temp_dir, "template.sh")
        with open(template_path, 'w') as f:
            f.write("#!/bin/bash\n"
                    "#SBATCH --nodes={nodes}\n"
                    "#SBATCH --partition={partition}\n")

        # Create executor
        executor = SlurmExecutor(root=temp_dir,
                                 template=template_path,
                                 slurm_config={"nodes": 1, "partition": "test"})

        yield executor


def test_executor_initialization(temp_executor):
    """Test that SlurmExecutor initializes correctly."""
    assert isinstance(temp_executor.root, pathlib.Path)
    assert temp_executor.slurm_config == {"nodes": 1, "partition": "test"}


def test_job_submission(temp_executor):
    """Test job submission process."""

    # Test function to submit
    def add(a, b):
        return a + b

    # Submit job
    job = temp_executor.submit(add, 5, 7)

    # Check job properties
    assert job.id == 12345, "Job ID should match mock value"
    # Check that job files were created
    assert (temp_executor.root / f"{job.file_prefix}.sh").exists()
    assert (temp_executor.root / f"{job.file_prefix}.py").exists()
    assert (temp_executor.root / f"{job.file_prefix}_function.pkl").exists()


def test_job_status_check():
    """Test job status checking."""
    # Create mock executor and job
    executor = mock.MagicMock()
    executor.root = pathlib.Path(tempfile.mkdtemp())

    job = Job(id=12345, status=JobStatus.PENDING, root=executor.root, file_prefix="test_job")

    # Mock the subprocess.run call
    with mock.patch('subprocess.run') as mock_run:
        mock_result = mock.MagicMock()
        mock_result.stdout = "RUNNING\n"
        mock_run.return_value = mock_result

        # Check status
        status = job.get_status()

        # Verify correct command was called
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "sacct"
        assert args[1] == "-j"
        assert args[2] == "12345"

        # Check status result
        assert status == JobStatus.RUNNING


def test_job_result_success():
    """Test successful job result retrieval."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock executor
        executor = mock.MagicMock()
        executor.root = pathlib.Path(temp_dir)
        executor.cleanup = False

        # Create job
        job = Job(id=12345, status=JobStatus.PENDING, root=executor.root, file_prefix="test_job")

        # Create mock result file
        result_value = 42
        result_path = executor.root / f"{job.file_prefix}_result.pkl"
        with open(result_path, 'wb') as f:
            cloudpickle.dump(result_value, f)

        # Get result
        assert job.result() == result_value


def test_job_result_failure():
    """Test handling of failed job."""
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock executor
        executor = mock.MagicMock()
        executor.root = pathlib.Path(temp_dir)
        executor.cleanup = False

        # Create job with FAILED status
        job = Job(id=12345, status=JobStatus.PENDING, root=executor.root, file_prefix="test_job")

        # Create mock error file
        error_path = executor.root / f"{job.file_prefix}_result.pkl.error"
        with open(error_path, 'w') as f:
            f.write("Test error message")

        # Try to get result, should raise RuntimeError with error details
        with pytest.raises(RuntimeError) as excinfo:
            job.result()

        error_str = str(excinfo.value)
        assert "Job 12345 failed with status" in error_str
        assert "Test error message" in error_str


def test_end_to_end_execution(temp_executor):
    """Test end-to-end job execution flow with mocks."""

    # Define test function
    def multiply(x, y):
        return x * y

    # Submit job
    job = temp_executor.submit(multiply, 6, 7)

    # Mock result file creation (normally done by the SLURM job)
    result_path = temp_executor.root / f"{job.file_prefix}_result.pkl"
    with open(result_path, 'wb') as f:
        cloudpickle.dump(42, f)

    # Mock get_status to return COMPLETED
    with mock.patch.object(job, 'get_status', return_value=JobStatus.COMPLETED):
        # Get result
        result = job.result()
        assert result == 42


def test_cleanup_files(temp_executor):
    """Test cleanup of job files."""
    # Submit a test job
    job = temp_executor.submit(lambda: None)

    # Get list of job files
    job_files = list(temp_executor.root.glob(f"{job.file_prefix}*"))
    assert len(job_files) > 0

    # Mock successful completion and result file
    result_path = temp_executor.root / f"{job.file_prefix}_result.pkl"
    with open(result_path, 'wb') as f:
        cloudpickle.dump(None, f)

    # Mock get_status to return COMPLETED
    with mock.patch.object(job, 'get_status', return_value=JobStatus.COMPLETED):
        # Get result with cleanup
        temp_executor.cleanup = True
        job.result()

        # Check that files were removed
        remaining_files = list(temp_executor.root.glob(f"{job.file_prefix}*"))
        assert len(remaining_files) == 0
