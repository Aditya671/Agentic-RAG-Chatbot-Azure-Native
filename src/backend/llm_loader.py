import os
import logging
from typing import Any, Optional, Union
from openai.lib.azure import AzureADTokenProvider

from azure.identity import get_bearer_token_provider, DefaultAzureCredential
from llama_index.core.llms.llm import LLM
from llama_index.llms.azure_inference import AzureAICompletionsModel
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.backend.ai_models import AIModelTypes
from src.backend.config import config, IndexConfig
from src.backend.credential_manager import CredentialManager


DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT = 10.0
logger = logging.getLogger(__name__)

def load_llm(
    model: AIModelTypes,
    index_name: str,
    temperature: Optional[float] = DEFAULT_TEMPERATURE,
    timeout: Optional[float] = DEFAULT_TIMEOUT,  # timeout in seconds
    azure_openai_use_azure_ad: bool = True,  # New parameter for Managed Identity
    additional_kwargs = {},
    callback_manager: Optional[Any] = None,
    # azure_openai_ad_token_provider: Optional[AzureADTokenProvider] = None,  # New parameter
    # azure_ai_endpoint: Optional[str] = None, # For AzureAICompletionsModel (e.g. DeepSeek)
    # azure_ai_credential: Optional[Any] = None, # For AzureAICompletionsModel - kept for flexibility, though deepseek uses key name
    use_azure: bool = True,
) -> LLM:
    """
    Load and instantiate an LLM instance based on the provided parameters.

    - Uses AzureOpenAI if 'azure_openai_endpoint', 'azure_openai_api_version', and 'azure_openai_use_azure_ad' are provided.
    - Otherwise, falls back to OpenAI.
    The 'timeout' parameter limits the response time (in seconds) for API calls.

    Returns:
        An instantiated LLM model.
    Raises:
        ValueError: If required configurations for a selected model are missing.
    """
    logger.info(f"Loading LLM model: {model} with temperature: {temperature}")
    index_config: Optional[IndexConfig] = config.indexes.get(index_name)
    if index_config is None:
        raise ValueError(f"Configuration for index '{index_name}' not found.")
    
    key_vault_config = index_config.key_vault
    if key_vault_config is None or key_vault_config.get("url") is None:
        raise ValueError(f"Key vault URL configuration is missing for index '{index_name}'.")
    credential_manager = CredentialManager(key_vault_url=key_vault_config["url"])
    if use_azure:
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        azure_openai_endpoint = index_config.llms.get('aoai').get('endpoint-east-us-2')
        azure_openai_api_version = index_config.llms.get('aoai').get('api-version-east-us-2')
        additional_kwargs = additional_kwargs
        llm = AzureOpenAI(
            model=model.value,
            engine=model.value,  # Assume deployment name matches model name.
            temperature=temperature,
            azure_ad_token_provider=token_provider,
            use_azure_ad=azure_openai_use_azure_ad,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_openai_api_version,
            request_timeout=float(timeout),
            additional_kwargs=additional_kwargs,
            callback_manager=callback_manager
        )
        logger.info(f"LLM model loaded successfully: {model}")
        return llm
    else:
        if model == AIModelTypes.O4_MINI_HIGH:
            llm = OpenAI(
                temperature=temperature,
                model=AIModelTypes.O4_MINI.value,
                reasoning_effort='high',
                api_key=credential_manager.get_secret('openai_api_key_name') or os.environ.get('OPENAI_API_KEY', None),
                request_timeout=float(timeout),  # Also pass timeout for consistency.
                additional_kwargs=additional_kwargs,
                callback_manager=callback_manager
            )
        else:
            llm = OpenAI(
                temperature=temperature,
                api_key=credential_manager.get_secret('openai_api_key_name') or os.environ.get('OPENAI_API_KEY', None),
                model=model.value,
                request_timeout=float(timeout),  # Also pass timeout for consistency.
                additional_kwargs=additional_kwargs,
                callback_manager=callback_manager
            )

        logger.info(f"LLM model loaded successfully: {model}")
        return llm


def load_embed(
    index_name: str,
    azure_openai_use_azure_ad: bool = True,  # New parameter for Managed Identity
    use_azure: bool = True,
    callback_manager: Optional[Any] = None,
) -> Union[AzureOpenAIEmbedding, OpenAIEmbedding]:
    """
    Load and instantiate an Embeddings instance based on the index profile name.

    Args:
        index_name: Name of the index profile in config.yaml
        azure_openai_endpoint: Azure OpenAI endpoint
        azure_openai_api_version: Azure OpenAI API version
        azure_openai_use_azure_ad: Whether to use Azure AD for authentication
        use_azure: Whether to use Azure OpenAI
        
    """
    index_config: Optional[IndexConfig] = config.indexes.get(index_name)
    model = index_config.embed.get("model")
    key_vault_config = index_config.key_vault
    if key_vault_config is None or key_vault_config.get("url") is None:
        raise ValueError(f"Key vault URL configuration is missing for index '{index_name}'.")
    credential_manager = CredentialManager(key_vault_url=key_vault_config["url"])
    if index_config is None:
        raise ValueError(f"Configuration for index '{index_name}' not found.")
    # Determine if using Azure OpenAI or regular OpenAI
    if use_azure:
        azure_openai_endpoint = index_config.llms.get('aoai').get('endpoint-east-us-2')
        azure_openai_api_version = index_config.llms.get('aoai').get('api-version-east-us-2')
        
        # If using Azure OpenAI, use the AzureOpenAIEmbedding model with managed identity
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")
        embed_model = AzureOpenAIEmbedding(
            model=model,
            deployment_name=model,    # Assume deployment name matches model name.
            azure_ad_token_provider=token_provider,
            use_azure_ad=azure_openai_use_azure_ad,
            azure_endpoint=azure_openai_endpoint,
            api_version=azure_openai_api_version,
            callback_manager=callback_manager
        )
        return embed_model
    else:
        embed_model = OpenAIEmbedding(
            model=model,
            api_key=credential_manager.get_secret('openai_api_key_name') or os.environ.get('OPENAI_API_KEY', None),
            temperature=0.1,
            request_timeout=float(10.0),
            callback_manager=callback_manager
        )
        return embed_model