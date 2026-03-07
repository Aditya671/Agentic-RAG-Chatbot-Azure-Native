import logging
import os
from e2b import Sandbox

logger = logging.getLogger(__name__)

class CodeInterpreterSandbox:
    """
    A secure sandbox for executing Python code using E2B.
    """

    def __init__(self, api_key: str = None):
        """
        Initializes the E2B sandbox.

        Args:
            api_key (str, optional): E2B API key. Defaults to E2B_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("E2B_API_KEY")
        if not self.api_key:
            logger.warning("E2B_API_KEY not found. Code interpreter will not be available.")
            self.sandbox = None
        else:
            # The sandbox is created on-demand to avoid long startup times.
            self.sandbox = None
            logger.info("CodeInterpreterSandbox initialized. Sandbox will be created on first use.")

    def run_python(self, code: str) -> str:
        """
        Executes a block of Python code in a secure sandbox.

        Args:
            code (str): The Python code to execute.

        Returns:
            str: The stdout, stderr, or error message from the execution.
        """
        if self.sandbox is None and self.api_key:
            try:
                self.sandbox = Sandbox(api_key=self.api_key)
            except Exception as e:
                error_message = f"Failed to create E2B sandbox: {e}"
                logger.error(error_message)
                return error_message
        elif self.sandbox is None:
            return "E2B sandbox is not available. Please set the E2B_API_KEY."

        try:
            logger.info(f"Executing code in sandbox:\n{code}")
            # Escape single quotes in the code for the shell command
            escaped_code = code.replace("'", "'\"'\"'")
            process = self.sandbox.process.start(f"python -c '{escaped_code}'")
            process.wait()

            output = process.stdout or process.stderr
            return f"Execution finished. Output:\n{output}"
        except Exception as e:
            logger.error(f"Error executing code in sandbox: {e}")
            return f"An unexpected error occurred during execution: {e}"

    def close(self):
        """
        Closes the sandbox to release resources.
        """
        if self.sandbox:
            self.sandbox.close()
            logger.info("E2B sandbox closed.")