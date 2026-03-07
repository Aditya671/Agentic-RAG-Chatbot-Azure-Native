import logging
import os
from typing import List

from llama_index.core import StorageContext, KnowledgeGraphIndex, Document
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.core.llms import LLM
from llama_index.core.settings import Settings
from llama_index.graph_stores.nebula import NebulaGraphStore

logger = logging.getLogger(__name__)

class GraphRAGSystem:
    """
    A system for building and querying a Knowledge Graph for advanced RAG.

    This system uses LlamaIndex's KnowledgeGraphIndex to extract entities and
    relationships from documents, storing them in a graph structure. This enables
    multi-hop reasoning and querying complex relationships within the data.
    """

    def __init__(self, llm: LLM, embed_model):
        """
        Initializes the Graph RAG system.

        Args:
            llm (LLM): The language model for graph construction and querying.
            embed_model: The embedding model for node embeddings.
        """
        self.llm = llm
        self.embed_model = embed_model
        try:
            # Attempt to connect to NebulaGraph for a persistent store
            self.graph_store = NebulaGraphStore(
                space_name=os.environ["NEBULA_SPACE_NAME"],
                edge_types=["relationship"],
                rel_prop_names=["relationship"],
                tags=["entity"],
            )
            logger.info(f"GraphRAGSystem initialized with persistent NebulaGraphStore, connected to space '{os.environ['NEBULA_SPACE_NAME']}'.")
        except Exception as e:
            # Fallback to in-memory store if Nebula is not available
            logger.warning(f"Failed to connect to NebulaGraph: {e}. Falling back to in-memory SimpleGraphStore. Ensure Nebula is running and environment variables are set for persistence.")
            self.graph_store = SimpleGraphStore()
        self.storage_context = StorageContext.from_defaults(graph_store=self.graph_store)
        self.index = None

    def build_graph_from_documents(self, documents: List[Document]):
        """
        Builds or updates the knowledge graph from a list of documents.

        Args:
            documents (List[Document]): The documents to process.
        """
        if not documents:
            logger.warning("No documents provided to build the graph. Skipping.")
            return

        logger.info(f"Building knowledge graph from {len(documents)} documents.")
        original_llm, original_embed_model = Settings.llm, Settings.embed_model
        Settings.llm, Settings.embed_model = self.llm, self.embed_model

        try:
            self.index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=self.storage_context,
                max_triplets_per_chunk=2,
                include_embeddings=True,
            )
            logger.info("Knowledge graph built successfully.")
        except Exception as e:
            logger.error(f"Failed to build knowledge graph: {e}")
            raise
        finally:
            Settings.llm, Settings.embed_model = original_llm, original_embed_model

    def as_query_engine(self):
        """
        Returns a query engine for the knowledge graph.
        """
        if self.index:
            return self.index.as_query_engine(
                include_text=False,
                response_mode="tree_summarize",
                embedding_mode="hybrid",
                similarity_top_k=5,
            )
        logger.warning("Graph index not built yet. Cannot create query engine.")
        return None

    def query(self, query_text: str) -> str:
        """
        Performs a query against the knowledge graph.
        """
        query_engine = self.as_query_engine()
        if query_engine:
            response = query_engine.query(query_text)
            return str(response)
        return "Knowledge graph is not available for querying."