# 📌 Upcoming Tasks for Cloud Storage Enhancements

## 1. Upload Methods
- **upload_file(local_path, destination_name)**  
  Upload a local file to Blob/S3 storage.  
  Useful for pipelines where you generate reports or artifacts locally and then push them back to cloud storage.

- **upload_stream(blob_stream, destination_name)**  
  Upload directly from a `BlobStream` object (or `BytesIO`) without needing a local file.  
  Handy for in-memory transformations.

---

## 2. Delete Methods
- **delete_blob(name)**  
  Remove a specific blob/object.  
  Important for cleanup tasks or managing temporary files.

- **delete_prefix(prefix)**  
  Bulk delete all blobs/objects under a given prefix (like a folder).  
  Useful for resetting environments or clearing old data.

---

## 3. List & Filter Methods
- **list_files(prefix=None, extension=None)**  
  Return a list of blob/object names that match a prefix and/or extension.  
  Helpful for enumerating available files before deciding which to fetch.

- **list_latest(n=5, prefix=None, extension=None)**  
  Return the latest *n* files instead of just one.  
  Useful when you want a batch of recent files (e.g., last 5 reports).

---

## 4. Metadata Retrieval
- **get_metadata(name)**  
  Fetch metadata (size, content type, last modified, etag) without downloading the file.  
  Saves bandwidth when you only need info.

---

## 5. Presigned URL / SAS Token Generation
- **generate_presigned_url(name, expiry_seconds=3600)**  
  Create a temporary URL for secure sharing or client-side download without exposing credentials.  
  - AWS: Presigned URL  
  - Azure: SAS token  

---

## 6. Archive / Versioning Helpers
- **archive_latest(prefix, extension, archive_prefix)**  
  Move or copy the latest file into an archive folder/prefix.  
  Useful for workflows where you want to keep history but still fetch “latest.”

---

## ✨ Why These Matter
- **Upload** → closes the loop (not just retrieval).  
- **Delete** → keeps storage clean.  
- **List/Filter** → gives visibility into what’s available.  
- **Metadata** → lightweight checks without downloads.  
- **Presigned URLs / SAS tokens** → secure sharing with external systems.  
- **Archive/versioning** → supports data lifecycle management.