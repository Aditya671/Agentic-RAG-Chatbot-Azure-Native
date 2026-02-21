import io
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Iterable, Any
from json import loads
import boto3
from botocore.exceptions import ClientError


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


class S3FileRetriever:
    def __init__(self, bucket_name: str, aws_access_key_id: str = None, aws_secret_access_key: str = None, region_name: str = None):
        try:
            self.s3_client = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                region_name=region_name,
            )
            self.bucket_name = bucket_name
            # Validate bucket exists
            self.s3_client.head_bucket(Bucket=bucket_name)
        except ClientError as e:
            raise ValueError(f"Bucket '{bucket_name}' not accessible: {e}")

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

    def _iter_objects(self, prefix: Optional[str]) -> Iterable:
        paginator = self.s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix or ""):
            for obj in page.get("Contents", []):
                yield obj

    def _pick_latest_object(self, objects: Iterable, extension: str, prefer_name_date: bool, date_regex: str, date_format: str):
        extension = extension.lower()
        best_obj = None
        best_key = None

        for obj in objects:
            name_lower = obj["Key"].lower()
            if not name_lower.endswith(extension):
                continue

            if prefer_name_date:
                dt_from_name = self._parse_date_from_name(
                    obj["Key"].rsplit("/", 1)[-1], date_regex, date_format
                )
            else:
                dt_from_name = None

            if dt_from_name:
                primary = dt_from_name
                secondary = self._to_utc_naive(obj.get("LastModified"))
                key = (1, primary, secondary)
            else:
                primary = self._to_utc_naive(obj.get("LastModified"))
                key = (0, primary)

            if best_key is None or key > best_key:
                best_key = key
                best_obj = obj

        return best_obj

    def get_latest_file_stream(
        self,
        prefix: Optional[str] = None,
        extension: str = ".csv",
        prefer_name_date: bool = True,
        date_regex: str = r"(\d{8})",
        date_format: str = "%Y%m%d",
    ) -> BlobStream:
        objects = list(self._iter_objects(prefix))
        if not objects:
            raise FileNotFoundError(f"No objects found with prefix='{prefix or ''}'")

        latest = self._pick_latest_object(
            objects=objects,
            extension=extension,
            prefer_name_date=prefer_name_date,
            date_regex=date_regex,
            date_format=date_format,
        )
        if not latest:
            raise FileNotFoundError(f"No matching '*{extension}' objects found under prefix='{prefix or ''}'")

        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=latest["Key"])
        content = response["Body"].read()
        stream = io.BytesIO(content)

        return BlobStream(
            name=latest["Key"],
            size=latest.get("Size", len(content)),
            content_type=response.get("ContentType"),
            last_modified=latest.get("LastModified"),
            etag=latest.get("ETag"),
            stream=stream,
        )

    def get_object(self, file_name: str) -> BlobStream:
        response = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_name)
        content = response["Body"].read()
        return BlobStream(
            name=file_name,
            size=response.get("ContentLength", len(content)),
            content_type=response.get("ContentType"),
            last_modified=response.get("LastModified"),
            etag=response.get("ETag"),
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
        extension: str = ".pdf",
    ) -> str:
        """Helper method to find the latest file of any extension, download it, and save it locally."""
        blob_stream = self.get_latest_file_stream(prefix=prefix, extension=extension)

        filename = os.path.basename(blob_stream.name)
        destination = os.path.join(local_directory, filename)

        self.download_to_file(blob_stream, destination)
        return destination