from datetime import datetime
from typing import Union
import logging
import nest_asyncio

from azure.identity import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.indexes import SearchIndexClient
# LlamaIndex core imports
from llama_index.core import ( Settings, StorageContext, VectorStoreIndex )
# from llama_index.core.query_pipeline import QueryPipeline as QP #Deprecated since llamaIndex version 0.14.3
# LlamaIndex provider-specific imports
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.azureaisearch import AzureAISearchVectorStore


credential = DefaultAzureCredential()

# Apply nest_asyncio to allow nested loops
nest_asyncio.apply()
CURRENT_DATE = datetime.now().strftime("%Y-%m-%d")
DEFAULT_TOP_K = 25
DEFAULT_TEMPERATURE = 0.1

logger = logging.getLogger(__name__)


def initialize_index(
        search_index_name: str,
        llm,
        embed_model: Union[AzureOpenAIEmbedding, OpenAIEmbedding],
        embed_size: int,    # the embed_model class doesn't capture the dimension info so it has to be passed in
        search_service_endpoint: str,
        search_service_credential,
        use_azure: bool = True,
        **kwargs
    ) -> VectorStoreIndex:
    """
    Initialize existing index to search using either Azure OpenAI or OpenAI with Azure Search resources.
    
    Args:
        search_index_name (str): Name of the index to create in Azure AI Search
        llm: Pre-configured LLM model instance (optional)
        embed_model: Pre-configured embedding model instance (optional)
        search_service_endpoint (str): Azure Search service endpoint
        search_service_credential: Azure Search service credential
        model_temperature (float): Temperature for the LLM. Defaults to 0.1
        **kwargs: Additional arguments including 'old_index' flag
    
    Returns:
        VectorStoreIndex: Initialized index
    """

    # Use index client to demonstrate creating an index
    if kwargs.get('aio') is None or kwargs.get('aio') is False:
        # Non-async client
        index_client = SearchIndexClient(
            endpoint=search_service_endpoint,
            credential=search_service_credential
        )
        
        base_params = {
            "search_or_index_client": index_client,
            "index_name": search_index_name,  # Include index_name for non-aio case
            "id_field_key": "id",
            "embedding_field_key": "embedding",
            "embedding_dimensionality": embed_size,
            "metadata_string_field_key": "metadata",
            "language_analyzer": "en.lucene",
            "vector_algorithm_type": "exhaustiveKnn"
        }
    else:
        # Async client
        index_client = SearchClient(
            endpoint=search_service_endpoint,
            index_name=search_index_name,
            credential=search_service_credential
        )
        
        base_params = {
            "search_or_index_client": index_client,
            # No index_name for aio case
            "id_field_key": "id",
            "embedding_field_key": "embedding",
            "embedding_dimensionality": embed_size,
            "metadata_string_field_key": "metadata",
            "language_analyzer": "en.lucene",
            "vector_algorithm_type": "exhaustiveKnn"
        }

    if kwargs.get('old_index') is None or kwargs.get('old_index') is False:
        base_params.update({
            "chunk_field_key": "chunk",
            "doc_id_field_key": "doc_id"
        })
        vector_store = AzureAISearchVectorStore(**base_params)
    else:
        base_params.update({
            "chunk_field_key": "content",
            "doc_id_field_key": "sourcepage",
            "filterable_metadata_field_keys": {
                "sourcefile": "sourcefile",
                "sourcepage": "sourcepage",
                "category": "category"
            },
            "searchable_fields": ["content", "filepath"],
            "hybrid_search": True
        })
        vector_store = AzureAISearchVectorStore(**base_params)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    Settings.llm = llm
    Settings.embed_model = embed_model
    return VectorStoreIndex.from_documents(
        [], storage_context=storage_context, llm=llm, embed_model=embed_model
    )
    