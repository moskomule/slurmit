import dataclasses
import enum
import logging
import pathlib
import re
import subprocess
import sys
import time
import uuid
from string import Template
from typing import Any, Callable, Generic, TypeVar, ParamSpec

import cloudpickle

P = ParamSpec('P')
R = TypeVar('R')


class JobStatus(enum.StrEnum):
    """Status of a SLURM job."""
    COMPLETED = enum.auto()
    PENDING = enum.auto()
    RUNNING = enum.auto()
    FAILED = enum.auto()
    CANCELLED = enum.auto()
    TIMEOUT = enum.auto()
    UNKNOWN = enum.auto()


@dataclasses.dataclass
class Job(Generic[R]):
    """
    Represents a job submitted to the SLURM cluster.

    Attributes:
        id: SLURM job ID.
        status: Current status of the job.
        root: Directory where job files are stored.
        file_prefix: Prefix for all files related to this job.
        cleanup: Whether to clean up job files after completion.
    """
    id: int
    status: JobStatus
    root: str | pathlib.Path
    file_prefix: str
    cleanup: bool = False

    def get_status(self) -> JobStatus:
        """
        Get the current status of the job from SLURM.

        Returns:
            Current status of the job.
        """
        try:
            result = subprocess.run(["sacct", "-j", str(self.id), "--format=State", "--noheader", "--parsable2"],
                                    check=True,
                                    capture_output=True,
                                    text=True
                                    )

            # Parse the output to get the job status
            status_str = result.stdout.strip().split('\n')[0].upper()

            # Map SLURM status to our JobStatus enum
            match status_str:
                case "COMPLETED":
                    return JobStatus.COMPLETED
                case "PENDING":
                    return JobStatus.PENDING
                case "RUNNING":
                    return JobStatus.RUNNING
                case "FAILED":
                    return JobStatus.FAILED
                case "CANCELLED":
                    return JobStatus.CANCELLED
                case "TIMEOUT" | "TIME_LIMIT":
                    return JobStatus.TIMEOUT
                case _:
                    return JobStatus.UNKNOWN

        except subprocess.CalledProcessError:
            return JobStatus.UNKNOWN

    def result(self) -> R:
        """
        Retrieves the result of the job. Blocks until the job completes.

        Returns:
            The return value of the function that was submitted.

        Raises:
            RuntimeError: If the job failed, was cancelled, or timed out.
        """
        # Wait for the job to complete
        while self.status in (JobStatus.PENDING, JobStatus.RUNNING):
            self.status = self.get_status()
            if self.status in (JobStatus.PENDING, JobStatus.RUNNING):
                time.sleep(60)  # Check every 60 seconds

        try:
            # Check if the job completed successfully
            if self.status != JobStatus.COMPLETED:
                error_path = self.root / f"{self.file_prefix}_result.pkl.error"
                error_message = f"Job {self.id} failed with status {self.status}"

                if error_path.exists():
                    with open(error_path, 'r') as f:
                        error_details = f.read()
                    error_message += f"\nError details:\n{error_details}"

                raise RuntimeError(error_message)

            # Load the result from the output file
            result_path = self.root / f"{self.file_prefix}_result.pkl"
            if not result_path.exists():
                raise RuntimeError(f"Job {self.id} completed but no result file was found at {result_path}")

            with open(result_path, 'rb') as f:
                result = cloudpickle.load(f)

            return result
        finally:
            # Clean up files if requested
            if self.cleanup:
                self._cleanup_files()

    def _cleanup_files(self):
        """Clean up temporary job files."""
        try:
            # Delete all files with this job's prefix
            for file in self.root.glob(f"{self.file_prefix}*"):
                file.unlink()
        except Exception as e:
            logging.warning(f"Failed to clean up job files: {e}")


class SlurmExecutor:
    """
    Executor for submitting jobs to a SLURM cluster.
    """

    def __init__(self,
                 root: str | pathlib.Path,
                 template: str | pathlib.Path,
                 slurm_config: dict[str, Any],
                 cleanup: bool = True,
                 submit_command: str = None,
                 pyton_path: str | pathlib.Path = None):
        """
        Initialize a SLURM executor.

        Args:
            root: Directory where job files will be stored.
            template: Path to the SLURM job template file.
            slurm_config: Configuration parameters for SLURM jobs.
            cleanup: Whether to clean up job files after completion.
            submit_command: If the submission is not simply `sbatch job.sh`, specify it as, e.g., "sbatch -l 1"
            pyton_path: Path to Python. If None, Python to run slurmit will be used.
        """

        self.root = pathlib.Path(root)
        self.slurm_config = slurm_config
        self.cleanup = cleanup

        # Check if SLURM is available
        self._check_slurm_available()

        # Create root directory if it doesn't exist
        self.root.mkdir(parents=True, exist_ok=True)

        # Check if template file exists
        template_path = pathlib.Path(template)
        if not template_path.exists():
            raise FileNotFoundError(f"Template file {template_path} not found")

        # Load template content
        with open(template_path, 'r') as f:
            self.template_content = f.read()

        self.submit_command = ["sbatch"] or submit_command.split(" ")
        self.python_path = sys.executable if pyton_path is None else pathlib.Path(pyton_path).resolve()

    @staticmethod
    def _check_slurm_available():
        """
        Check if SLURM is available on the system.

        Raises:
            RuntimeError: If SLURM is not available.
        """
        try:
            subprocess.run(["sbatch", "--version"], check=True, capture_output=True)
        except (subprocess.SubprocessError, FileNotFoundError):
            raise RuntimeError("SLURM is not available on this system. Make sure 'sbatch' is in your PATH.")

    def submit(self,
               f: Callable[P, R],
               *args: P.args,
               **kwargs: P.kwargs
               ) -> Job[R]:
        """
        Submit a function to be executed on the SLURM cluster.

        Args:
            f: Function to be executed.
            *args: Positional arguments to pass to the function.
            **kwargs: Keyword arguments to pass to the function.

        Returns:
            A Job object representing the submitted job.
        """
        # Generate a unique file prefix for this job
        file_prefix = f"{time.strftime('%Y%m%d%H%M')}_{uuid.uuid4().hex}"

        # Create paths for job files
        function_path = self.root / f"{file_prefix}_function.pkl"
        result_path = self.root / f"{file_prefix}_result.pkl"
        python_script_path = self.root / f"{file_prefix}.py"
        script_path = self.root / f"{file_prefix}.slurm"

        # Serialize function and arguments
        with open(function_path, 'wb') as f_out:
            cloudpickle.dump((f, args, kwargs), f_out)

        # Create a Python script to execute the function
        python_script = f"""
import cloudpickle
import traceback
import sys
import os

try:
    # Load function and arguments
    with open("{function_path}", "rb") as f:
        func, args, kwargs = cloudpickle.load(f)

    # Execute function
    result = func(*args, **kwargs)

    # Save result
    with open("{result_path}", "wb") as f:
        cloudpickle.dump(result, f)

    sys.exit(0)
except Exception as e:
    error_path = "{result_path}.error"
    with open(error_path, "w") as f:
        f.write(f"Error: {{str(e)}}\\n")
        f.write(traceback.format_exc())
    sys.exit(1)
   
"""

        with open(python_script_path, 'w') as f:
            f.write(python_script)

        # Create SLURM job script by replacing placeholders in the template
        template = Template(self.template_content)
        slurm_script = template.safe_substitute(**self.slurm_config)

        # Add command to execute the Python script
        slurm_script += f"{self.python_path} {python_script_path}\n"

        # Write SLURM job script to file
        with open(script_path, 'w') as f:
            f.write(slurm_script)

        # Submit job to SLURM
        try:
            result = subprocess.run(self.submit_command + [str(script_path)],
                                    check=True,
                                    capture_output=True,
                                    text=True
                                    )

            # Extract job ID from sbatch output (usually "Submitted batch job <job_id>")
            job_id_match = re.search(r"Submitted batch job (\d+)", result.stdout)

            if not job_id_match:
                raise RuntimeError(f"Failed to extract job ID from sbatch output: {result.stdout}")

            try:
                slurm_job_id = int(job_id_match.group(1))
            except (ValueError, IndexError) as e:
                raise RuntimeError(f"Failed to parse job ID as integer: {e}")

            # Create and return job object
            return Job(id=slurm_job_id,
                       status=JobStatus.PENDING,
                       root=self.root,
                       file_prefix=file_prefix,
                       cleanup=self.cleanup
                       )

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to submit job: {e.stderr}")
