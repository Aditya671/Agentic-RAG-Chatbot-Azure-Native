import io
import sys
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Iterable, Any
from json import loads
from azure.storage.blob import ContainerClient, BlobClient, BlobServiceClient
try:
    current_dir = os.path.dirname(__file__)
except NameError:
    current_dir = os.getcwd()  # fallback for environments where __file__ is not defined
sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))


@dataclass
class BlobStream:
    name: str
    size: int
    content_type: Optional[str]
    last_modified: Optional[datetime]
    etag: Optional[str]
    stream: io.BytesIO

    def to_bytes(self) -> bytes:
        pos = self.stream.tell()
        self.stream.seek(0)
        data = self.stream.read()
        self.stream.seek(pos)
        return data

    def to_json(self) -> Any:
        data = self.to_bytes().decode("utf-8")
        return loads(data)
    
    def to_str(self) -> str:
        return self.to_bytes().decode("utf-8")


class AzureBlobFileRetriever:
    def __init__(self,\
        container_client_service: ContainerClient,\
        connection_string: str = None, \
        container_name: str = None\
    ):
        try:
            if container_client_service.exists():
                self.container_client = container_client_service
            else:
                self.blob_service = BlobServiceClient.from_connection_string(conn_str=connection_string)
                self.container_client = self.blob_service.get_container_client(container=container_name)
        except Exception as e:
            raise ValueError("Either container_client_service or connection_string and container_name must be provided.")


    @staticmethod
    def _to_utc_naive(dt: Optional[datetime]) -> Optional[datetime]:
        if dt is None:
            return None
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt

    @staticmethod
    def _parse_date_from_name(name: str, date_regex: str, date_format: str) -> Optional[datetime]:
        m = re.search(date_regex, name)
        if not m:
            return None
        try:
            return datetime.strptime(m.group(1), date_format)
        except Exception:
            return None

    def _iter_blobs(self, prefix: Optional[str]) -> Iterable:
        return self.container_client.list_blobs(name_starts_with=prefix or "")

    def _pick_latest_blob(
        self,
        blobs: Iterable,
        extension: str,
        prefer_name_date: bool,
        date_regex: str,
        date_format: str,
    ):
        extension = extension.lower()
        best_blob = None
        best_key = None

        for b in blobs:
            name_lower = b.name.lower()
            if not name_lower.endswith(extension):
                continue

            if prefer_name_date:
                dt_from_name = self._parse_date_from_name(
                    b.name.rsplit("/", 1)[-1], date_regex, date_format
                )
            else:
                dt_from_name = None

            if dt_from_name:
                primary = dt_from_name
                secondary = self._to_utc_naive(b.last_modified)
                key = (1, primary, secondary)
            else:
                primary = self._to_utc_naive(b.last_modified)
                key = (0, primary)

            if best_key is None or key > best_key:
                best_key = key
                best_blob = b

        return best_blob

    def get_latest_file_stream(
        self,
        prefix: Optional[str] = None,
        extension: str = ".csv",
        prefer_name_date: bool = True,
        date_regex: str = r"(\d{8})",
        date_format: str = "%Y%m%d",
        max_concurrency: int = 4,
    ) -> BlobStream:
        blobs = list(self._iter_blobs(prefix))
        if not blobs:
            raise FileNotFoundError(f"No blobs found with prefix='{prefix or ''}'")

        latest = self._pick_latest_blob(
            blobs=blobs,
            extension=extension,
            prefer_name_date=prefer_name_date,
            date_regex=date_regex,
            date_format=date_format,
        )
        if not latest:
            raise FileNotFoundError(f"No matching '*{extension}' blobs found under prefix='{prefix or ''}'")

        blob_client: BlobClient = self.container_client.get_blob_client(latest.name)
        downloader = blob_client.download_blob(max_concurrency=max_concurrency)
        content = downloader.readall()
        stream = io.BytesIO(content)

        props = blob_client.get_blob_properties()
        return BlobStream(
            name=latest.name,
            size=getattr(props, "size", len(content)),
            content_type=getattr(props.content_settings, "content_type", None),
            last_modified=props.last_modified,
            etag=getattr(props, "etag", None),
            stream=stream,
        )

    def get_blob(self, file_name = '') -> BlobStream:
        blob_client = self.container_client.get_blob_client(file_name)
        downloaded_stream = blob_client.download_blob()
        content = downloaded_stream.readall()
        props = blob_client.get_blob_properties()
        return BlobStream(
            name=blob_client.blob_name,
            size=getattr(props, "size", len(content)),
            content_type=getattr(props.content_settings, "content_type", None),
            last_modified=props.last_modified,
            etag=getattr(props, "etag", None),
            stream=io.BytesIO(content),
        )

    def download_to_file(self, blob_stream: BlobStream, destination_path: str):
        """Writes a BlobStream object to the local file system."""
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        with open(destination_path, "wb") as f:
            f.write(blob_stream.to_bytes())
        print(f"Successfully saved to: {destination_path}")

    def get_and_save_latest(
        self, 
        local_directory: str, 
        prefix: Optional[str] = None, 
        extension: str = ".pdf"
    ) -> str:
        """
        Helper method to find the latest file of any extension, 
        download it, and save it locally.
        """
        blob_stream = self.get_latest_file_stream(prefix=prefix, extension=extension)
        
        # Construct local path
        filename = os.path.basename(blob_stream.name)
        destination = os.path.join(local_directory, filename)
        
        self.download_to_file(blob_stream, destination)
        return destination

# ---------------- Example usage ----------------
# if __name__ == "__main__":
#     index_name = os.getenv("INDEX_NAME", "aiim")
#     index_config = config_module.indexes.get(index_name)

#     if not index_config:
#         raise ValueError(f"No index configuration found for '{index_name}'")

#     try:
#         credential_manager = CredentialManager(key_vault_url=index_config.key_vault.get("url"))
#         connection_string_secret = credential_manager.client.get_secret(
#             index_config.storage_account.get('connection_string')
#         )
#         connection_string = connection_string_secret.value
#     except Exception as e:
#         raise RuntimeError(f"Failed to retrieve storage connection string: {e}")

#     blob_service = BlobServiceClient.from_connection_string(connection_string)

#     container_name = index_config.storage_account.get('container_name')
#     container_client = blob_service.get_container_client(container_name)
    
#     agent = AzureBlobFileRetriever(container_client=container_client)

#     try:
#         blob_stream = agent.get_latest_file_stream(prefix="your_file", extension=".csv")
#         print(f"Downloaded: {blob_stream.name}, size={blob_stream.size}")
#     except FileNotFoundError as e:
#         print(f"Error: {e}")

#     # Download the latest PDF
#     pdf_path = agent.get_and_save_latest(local_directory="./downloads", extension=".pdf")

#     # Download the latest Word Document
#     docx_path = agent.get_and_save_latest(local_directory="./downloads", extension=".docx")

#     import pandas as pd

#     # For CSV
#     csv_stream = agent.get_latest_file_stream(extension=".csv")
#     df_csv = pd.read_csv(io.BytesIO(csv_stream.to_bytes()))

#     # For Excel
#     xlsx_stream = agent.get_latest_file_stream(extension=".xlsx")
#     df_xlsx = pd.read_excel(io.BytesIO(xlsx_stream.to_bytes()))