import os
import yaml
import logging
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from enum import Enum

# Constants in UPPER_CASE
DEFAULT_ENVIRONMENT: str = "local"
VALID_ENVIRONMENTS = {"local", "development", "uat", "staging", "production"}

# Enum class in PascalCase
class Environment(str, Enum):
    LOCAL = "local"
    LOCAL_EMULATOR = 'local_emulator'
    DEVELOPMENT = "development"
    UAT = "uat"
    STAGING = "staging"
    PRODUCTION = "production"

class IndexConfig:
    """
    Represents the index-specific configuration.
    This ties together the Azure AI Search, storage, embed, and RAG settings.
    """
    def __init__(self, name: str, settings: Dict[str, Any]) -> None:
        self.name = name
        self.settings = settings

    @property
    def azure_ai_search(self) -> Dict[str, Any]:
        return self.settings.get("azure_ai_search", {})

    @property
    def storage_account(self) -> Dict[str, Any]:
        return self.settings.get("storage_account", {})

    @property
    def embed(self) -> Dict[str, Any]:
        return self.settings.get("embed", {})

    @property
    def rag(self) -> Dict[str, Any]:
        return self.settings.get("rag", {})
    
    @property
    def key_vault(self) -> Dict[str, Any]:
        return self.settings.get("key_vault", {})
    
    @property
    def di(self) -> Dict[str, Any]:
        """Get the Document Intelligence configuration."""
        return self.settings.get("di", {})
    
    @property
    def llms(self) -> Dict[str, Any]:
        """Get the LLM configurations."""
        return self.settings.get("llms", {})

    @property
    def dev_cosmos_db(self) -> Dict[str, Any]:
        """Get the Cosmos DB Dev configuration."""
        return self.settings.get("dev_cosmos_db", {})
    
    @property
    def uat_cosmos_db(self) -> Dict[str, Any]:
        """Get the Cosmos DB Uat configuration."""
        return self.settings.get("uat_cosmos_db", {})
    
    @property
    def prod_cosmos_db(self) -> Dict[str, Any]:
        """Get the Cosmos DB Prod configuration."""
        return self.settings.get("prod_cosmos_db", {})
    
    @property
    def ai_service(self) -> Dict[str, Any]:
        """Get the Cosmos DB Prod configuration."""
        return self.settings.get("ai_service", {})

# Class in PascalCase
class Config:
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._config_path: str = str(config_path or os.getenv("CONFIG_PATH", "./config.yml"))
        logging.info(f"Config path set to: {self._config_path}")
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Absolute config path: {os.path.abspath(self._config_path)}")
        self._config: Optional[Dict[str, Any]] = None
        self._load_env_if_local()
    
    def _load_env_if_local(self) -> None:
        """Load environment variables from .env file if in local development mode."""
        if self.environment == Environment.LOCAL:
            # Prefer .env.local if it exists, otherwise .env
            env_file = ".env.local" if os.path.exists(".env.local") else ".env"
            load_dotenv(env_file)
            logging.debug(f"Loaded local environment file: {env_file}")
    
    @property
    def environment(self) -> Environment:
        """Get the current environment."""
        env_str = os.getenv("ENVIRONMENT", "local").lower()
        try:
            return Environment(env_str)
        except ValueError:
            logging.warning(f"Invalid ENVIRONMENT value '{env_str}'. Defaulting to LOCAL.")
            return Environment.LOCAL
    
    @property
    def is_local(self) -> bool:
        """Check if running in local development environment."""
        return self.environment == Environment.LOCAL
    
    @property
    def is_cloud(self) -> bool:
        """Check if running in any cloud environment."""
        return self.environment in {
            Environment.DEVELOPMENT,
            Environment.STAGING,
            Environment.PRODUCTION
        }
    
    @property
    def key_vault_url(self) -> Optional[str]:
        """
        Retrieve the Azure Key Vault URL from the environment variable or
        from the YAML configuration file.
        """
        azure_url = os.getenv("AZURE_KEY_VAULT_URL")
        if azure_url:
            return azure_url
        return self._get_config().get("azure", {}).get("key_vault", {}).get("url")
    
    def _get_config(self) -> Dict[str, Any]:
        """
        Load and return the configuration from a YAML file.
        """
        if self._config is None:
            try:
                logging.info(f"Attempting to load config from: {self._config_path}")
                with open(self._config_path, "r") as f:
                    loaded_config = yaml.safe_load(f) or {}
                    if self.environment == Environment.LOCAL:
                        logging.info(f"Loaded config: {loaded_config}")
                    self._config = loaded_config
            except FileNotFoundError:
                logging.warning(f"Configuration file not found at '{self._config_path}'. Using empty config.")
                self._config = {}
            except yaml.YAMLError as err:
                logging.error(f"Error parsing YAML file '{self._config_path}': {err}. Using empty config.")
                self._config = {}
        return self._config

    @property
    def indexes(self) -> Dict[str, IndexConfig]:
        """
        Returns a dictionary of index configurations, keyed by index name.
        Supports both YAML config and environment variables.
        """
        # First try to get from YAML config
        indexes_config = self._get_config().get("indexes", {})
        return {name: IndexConfig(name, settings) for name, settings in indexes_config.items()}

    @property
    def llms(self) -> Dict[str, Any]:
        """
        Returns the LLM configurations from the config file.
        """
        return self._get_config().get("llms", {})

    def get_llm_config(self, model_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific LLM model.
        
        Args:
            model_name: The name of the LLM model (e.g., 'deepseek-r1')
            
        Returns:
            Dictionary containing the model's configuration
            
        Raises:
            KeyError: If the requested model configuration is not found
        """
        llm_config = self.llms.get(model_name)
        if not llm_config:
            raise KeyError(f"Configuration for LLM model '{model_name}' not found in config file")
        return llm_config

    @property
    def document_intelligence_api_key_name(self) -> Optional[str]:
        """Get the Document Intelligence API key name from Key Vault."""
        return self._get_config().get("azure", {}).get("key_vault", {}).get("document_intelligence_api_key_name")

    @property
    def document_intelligence_endpoint(self) -> Optional[str]:
        """Get the Document Intelligence endpoint."""
        return self._get_config().get("azure", {}).get("key_vault", {}).get("document_intelligence_endpoint")

    @property
    def openai_api_key_name(self) -> Optional[str]:
        """Get the OpenAI API key name from Key Vault."""
        return self._get_config().get("azure", {}).get("key_vault", {}).get("openai_api_key_name")

    @property
    def key_vault_config(self) -> Dict[str, Any]:
        """Get the complete Key Vault configuration."""
        return self._get_config().get("azure", {}).get("key_vault", {})

    @property
    def cosmos_db_uri(self) -> Optional[str]:
        """Get the Cosmos Db Key from key Vault."""
        return self._get_config().get("azure", {}).get("key_vault", {}).get("uri")
    
    @property
    def ai_service(self) -> Dict[str, Any]:
        """
        Returns the AI Service configurations from the config file.
        """
        return self._get_config().get("ai_service", {})

# Module-level instance in lowercase with underscores
config: Config = Config()