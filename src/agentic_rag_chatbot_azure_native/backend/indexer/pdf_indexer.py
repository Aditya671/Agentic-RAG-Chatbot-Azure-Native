import os
import hashlib
import uuid
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
from llama_index import Document, GPTVectorStoreIndex, ServiceContext
from llama_index.embeddings import AzureOpenAIEmbedding
from llama_index.vector_stores import AzureAISearchVectorStore
from app_logger import setup_logger

# Initialize logger
logger, _ = setup_logger(name="pdf-indexer")

# Configurable parameters
CHUNK_SIZE = 1000          # characters (or tokens depending on strategy)
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 16

# Environment variables required
# AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_SEARCH_SERVICE_NAME, AZURE_SEARCH_API_KEY, AZURE_SEARCH_INDEX_NAME

def compute_checksum(file_path: str) -> str:
    """Compute SHA256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def extract_text_from_pdf(file_path: str) -> Tuple[str, int]:
    """Extract text from PDF using PyMuPDF. Returns full text and page count."""
    doc = fitz.open(file_path)
    text_parts = []
    for page in doc:
        text = page.get_text("text")
        text_parts.append(text)
    full_text = "\n".join(text_parts)
    return full_text, len(doc)

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Tuple[str, int, int]]:
    """
    Chunk text into overlapping windows.
    Returns list of tuples: (chunk_text, start_offset, end_offset)
    """
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        if end == text_len:
            break
        start = max(end - overlap, end) - (0 if end - overlap < 0 else 0)
        # Move start forward by chunk_size - overlap
        start = start + (chunk_size - overlap) if start == 0 else end - overlap
    return chunks

def build_metadata_for_doc(file_path: str, checksum: str) -> Dict:
    stat = os.stat(file_path)
    return {
        "doc_id": str(uuid.uuid5(uuid.NAMESPACE_URL, file_path)),
        "source_path": file_path,
        "filename": os.path.basename(file_path),
        "checksum": checksum,
        "uploaded_date": datetime.utcnow().isoformat() + "Z",
        "last_modified": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z",
        "file_size": stat.st_size,
        "mime_type": "application/pdf",
        "index_version": "v1",
    }

def create_documents_from_pdf(file_path: str) -> Tuple[List[Document], Dict]:
    checksum = compute_checksum(file_path)
    text, page_count = extract_text_from_pdf(file_path)
    metadata = build_metadata_for_doc(file_path, checksum)
    metadata["page_count"] = page_count

    chunks = chunk_text(text)
    docs = []
    for i, (chunk_text, start, end) in enumerate(chunks):
        chunk_id = f"{metadata['doc_id']}::chunk::{i}"
        chunk_meta = metadata.copy()
        chunk_meta.update({
            "chunk_id": chunk_id,
            "chunk_start_offset": start,
            "chunk_end_offset": end,
            "chunk_start_page": None,  # optional: compute page mapping if needed
            "chunk_end_page": None,
            "indexed_at": datetime.utcnow().isoformat() + "Z",
        })
        docs.append(Document(text=chunk_text, metadata=chunk_meta))
    return docs, metadata

def init_embedding_and_vectorstore() -> Tuple[AzureOpenAIEmbedding, AzureAISearchVectorStore, ServiceContext]:
    # Azure OpenAI Embedding init
    embedding = AzureOpenAIEmbedding(
        deployment="text-embedding-3-large",  # adjust to your deployment name
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    )

    # Azure AI Search vector store init
    azure_search_service = os.environ.get("AZURE_SEARCH_SERVICE_NAME")
    azure_search_key = os.environ.get("AZURE_SEARCH_API_KEY")
    azure_index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME")

    vector_store = AzureAISearchVectorStore(
        service_name=azure_search_service,
        api_key=azure_search_key,
        index_name=azure_index_name,
        embedding=embedding,
    )

    # ServiceContext for llama-index
    service_context = ServiceContext.from_defaults(embed_model=embedding)
    return embedding, vector_store, service_context

def upsert_documents_to_index(docs: List[Document], vector_store: AzureAISearchVectorStore, service_context: ServiceContext):
    """
    Create or update index with documents. Uses batching for embeddings.
    """
    logger.info(f"Upserting {len(docs)} chunks to vector store")
    index = GPTVectorStoreIndex.from_documents(docs, service_context=service_context, vector_store=vector_store)
    # If you prefer incremental upsert, use vector_store.upsert or index.save_to_disk depending on store API
    logger.info("Upsert complete")

def index_pdf(file_path: str, force_reindex: bool = False):
    """
    High-level function to index a single PDF.
    - Skips indexing if checksum unchanged unless force_reindex is True.
    """
    logger.info(f"Indexing file: {file_path}")
    checksum = compute_checksum(file_path)
    # TODO: check existing metadata store (DB) for checksum to skip reindex
    # Example: existing = metadata_db.get_by_doc_id(doc_id); if existing and existing['checksum'] == checksum: skip

    docs, metadata = create_documents_from_pdf(file_path)
    embedding, vector_store, service_context = init_embedding_and_vectorstore()
    upsert_documents_to_index(docs, vector_store, service_context)

    # Persist metadata to metadata DB (CosmosDB, MongoDB, or Azure Table)
    # metadata_db.upsert(metadata)
    logger.info(f"Indexed {len(docs)} chunks for doc_id {metadata['doc_id']}")