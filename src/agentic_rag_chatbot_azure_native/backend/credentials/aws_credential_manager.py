import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

class AWSCredentialManager:
    def __init__(self, secret_name: str = None, region_name: str = "us-east-1"):
        """
        Initializes the AWS credential manager.

        Parameters:
            secret_name (str): The name of the AWS Secrets Manager secret to use.
                               If None, only environment variables will be used.
            region_name (str): AWS region where the secret is stored.
        """
        self.secret_name = secret_name
        self.region_name = region_name
        if secret_name:
            self.client = self.get_client()
        else:
            self.client = None

    @staticmethod
    def get_session() -> boto3.session.Session:
        """
        Determines the appropriate AWS credential source:
        - Local development: Uses environment variables or AWS CLI profile.
        - Production: Uses IAM role (EC2, ECS, Lambda, etc.).

        Returns:
            boto3.session.Session: An authenticated AWS session.
        """
        try:
            # Try default session (checks env vars, shared config, CLI, IAM role)
            return boto3.session.Session()
        except (NoCredentialsError, PartialCredentialsError):
            raise ValueError("No valid AWS credentials found. Configure AWS CLI or environment variables.")

    def get_client(self):
        """
        Creates a Secrets Manager client using the resolved session.
        """
        session = self.get_session()
        return session.client(service_name="secretsmanager", region_name=self.region_name)

    def get_secret(self, secret_name: str = None) -> str:
        """
        Retrieve a secret value from environment variables or AWS Secrets Manager.

        Parameters:
            secret_name (str): The name of the secret (e.g., "MY_APP_SECRET").
                               If None, uses the secret_name provided at initialization.

        Returns:
            str: The secret's value.

        Raises:
            ValueError: If the secret is not found in environment variables or Secrets Manager.
        """
        secret_name = secret_name or self.secret_name
        if not secret_name:
            raise ValueError("No secret name provided.")

        # Check environment variables first
        secret = os.environ.get(secret_name)
        if secret:
            return secret

        # If Secrets Manager is configured, try to get it from there
        if self.client:
            try:
                response = self.client.get_secret_value(SecretId=secret_name)
                if "SecretString" in response:
                    return response["SecretString"]
                elif "SecretBinary" in response:
                    return response["SecretBinary"].decode("utf-8")
            except self.client.exceptions.ResourceNotFoundException:
                raise ValueError(f"Secret '{secret_name}' not found in AWS Secrets Manager.")

        raise ValueError(f"Secret '{secret_name}' not found in environment variables or AWS Secrets Manager.")