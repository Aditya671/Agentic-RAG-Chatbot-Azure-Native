import asyncio
import os

from celery import Celery
from dotenv import load_dotenv

from backend.user_uploaded_file_indexer import UserUploadedFileIndexer

load_dotenv(override=True)

# Assumes Redis is running on localhost. Update the URLs for production.
celery_app = Celery(
    'tasks',
    broker=os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0"),
    backend=os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
)

@celery_app.task(name="tasks.index_files")
def index_files_task(file_list: list, root_dir: str, index_name: str, model: str, similarity_top_k: int):
    """
    Celery task to index uploaded files asynchronously.
    """
    # Note: The 'memory' object cannot be passed to a Celery task as it's not serializable.
    # The indexer is created fresh in the worker context.
    indexer = UserUploadedFileIndexer(
        root_dir=root_dir,
        index_name=index_name,
        model=model,
        memory=None,  # Memory is not available in the worker context
        similarity_top_k=similarity_top_k
    )
    result = asyncio.run(indexer.index_uploaded_files(file_list=file_list))
    return result