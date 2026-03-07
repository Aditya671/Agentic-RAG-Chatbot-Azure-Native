import os
from azure.identity import DefaultAzureCredential, AzureCliCredential
from azure.keyvault.secrets import SecretClient

class CredentialManager:
    def __init__(self, key_vault_url: str = None):
        """
        Initializes the credential manager.

        Parameters:
            key_vault_url (str): The URL of the Azure Key Vault to use. If None,
                                 only environment variables will be used.
        """
        self.key_vault_url = key_vault_url
        if key_vault_url:
            self.credential = self.get_credential()
            self.client = SecretClient(vault_url=key_vault_url, credential=self.credential)
        else:
            self.client = None

    @staticmethod
    def get_credential() -> DefaultAzureCredential:
        """
        This method ensures that:
        - In local development environments (e.g., when ENVIRONMENT is 'local' or 'local-emulator'),
        the credential chain supports developer tools like Azure CLI, Visual Studio Code, etc.
        - In production or cloud environments (e.g., App Services, Functions, AKS),
        the credential supports Managed Identity and multi-tenant access when needed.

        Returns the following depending upon the Environment Instance:
            DefaultAzureCredential: An instance of DefaultAzureCredential with appropriate configuration over Cloud Env.
            AzureCliCredential: An instance of AzureCliCredential with User Logged in to Azure Env over Azure Cli.
        Usage:
            Inside CredentialManager:
                self.credential = CredentialManager.get_credential()

            Or anywhere externally:
                from credential_manager import CredentialManager
                credential = CredentialManager.get_credential()
        """
        if (os.environ.get('ENVIRONMENT') or 'local') in ['local', 'local_emulator']:
            # For local development environments, use default tenant restrictions
            return AzureCliCredential(additionally_allowed_tenants=["*"])
        else:
            # For other environments (e.g., cloud), allow all tenants
            return DefaultAzureCredential()

    def get_secret(self, secret_name: str) -> str:
        """
        Retrieve a secret value from the environment variable or, if not present, from Azure Key Vault.

        Parameters:
            secret_name (str): The name of the secret (for example, "AZURE_OPENAI_API_KEY").
        
        Returns:
            str: The secret's value.
            
        Raises:
            ValueError: If the secret is not found in environment variables or if a Key Vault
                        was configured and the secret is missing there.
        """
        # Check environment variables first
        secret = os.environ.get(secret_name)
        if secret:
            return secret
        
        # If a Key Vault URL is configured, try to get it from there
        if self.client:
            secret_obj = self.client.get_secret(secret_name)
            if secret_obj and secret_obj.value:
                return secret_obj.value
        
        raise ValueError(f"Secret '{secret_name}' not found in environment variables or key vault.") 
