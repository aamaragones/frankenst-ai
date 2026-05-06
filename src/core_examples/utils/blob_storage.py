import json
import os
import tempfile
from functools import lru_cache
from typing import Any

from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient

from core_examples.utils.key_vault import get_secret


@lru_cache(maxsize=1)
def get_blob_service_client() -> BlobServiceClient:
    storage_account_name = get_secret("AZURE_BLOB_STORAGE_NAME")
    if not storage_account_name:
        raise ValueError("AZURE_BLOB_STORAGE_NAME secret not found")
    return BlobServiceClient(
        account_url=f"https://{storage_account_name}.blob.core.windows.net",
        credential=DefaultAzureCredential(),
    )


def ensure_container(container_name: str) -> None:
    blob_service_client = get_blob_service_client()

    try:
        blob_service_client.create_container(container_name)
    except ResourceExistsError:
        pass


def upload_file_to_blob(blob_path: str, content: str, container_name: str, overwrite: bool = True):
    blob_service_client = get_blob_service_client()
    ensure_container(container_name)

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)
    blob_client.upload_blob(content, overwrite=overwrite)

    return f"File '{blob_path}' uploaded successfully to container '{container_name}'."


def upload_text_blob(blob_path: str, content: str, container_name: str, overwrite: bool = True) -> str:
    return upload_file_to_blob(
        blob_path=blob_path,
        content=content,
        container_name=container_name,
        overwrite=overwrite,
    )


def upload_json_blob(
    blob_path: str,
    payload: dict[str, Any],
    container_name: str,
    overwrite: bool = True,
) -> str:
    if not isinstance(payload, dict):
        raise ValueError("The 'payload' value must be a JSON object.")

    return upload_text_blob(
        blob_path=blob_path,
        content=json.dumps(payload, ensure_ascii=False, indent=2),
        container_name=container_name,
        overwrite=overwrite,
    )


def delete_blob(blob_path: str, container_name: str) -> None:
    blob_service_client = get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)

    try:
        blob_client.delete_blob()
    except ResourceNotFoundError as exc:
        raise FileNotFoundError(
            f"Blob '{blob_path}' was not found in container '{container_name}'"
        ) from exc


def list_blobs(container_name: str, prefix: str | None = None) -> list[dict[str, Any]]:
    blob_service_client = get_blob_service_client()
    container_client = blob_service_client.get_container_client(container_name)

    try:
        blobs = container_client.list_blobs(name_starts_with=prefix)
    except ResourceNotFoundError as exc:
        raise FileNotFoundError(
            f"Container '{container_name}' was not found while listing blobs."
        ) from exc

    return [
        {
            "name": blob.name,
            "size": blob.size,
            "etag": blob.etag,
            "last_modified": blob.last_modified,
        }
        for blob in blobs
    ]

def load_text_from_blob(blob_path: str, container_name: str) -> str:
    blob_service_client = get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)

    try:
        download_stream = blob_client.download_blob()
        content = download_stream.readall().decode("utf-8")
        return content

    except ResourceNotFoundError:
        raise FileNotFoundError(
            f"Blob '{blob_path}' was not found in container '{container_name}'"
        )


def load_json_from_blob(blob_path: str, container_name: str) -> dict[str, Any]:
    """Load a JSON blob and return it as a Python mapping."""

    try:
        content = load_text_from_blob(blob_path=blob_path, container_name=container_name)
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Blob '{blob_path}' in container '{container_name}' does not contain valid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Blob '{blob_path}' in container '{container_name}' must contain a JSON object at the root."
        )

    return payload


def download_blob_to_temp_file(blob_path: str, container_name: str) -> str:
    """Download any blob from Azure Blob Storage into a local temporary file."""

    blob_service_client = get_blob_service_client()
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_path)

    try:
        filename = os.path.basename(blob_path)
        download_stream = blob_client.download_blob()

        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, filename)

        with open(temp_path, "wb") as temp_file:
            temp_file.write(download_stream.readall())

        return temp_path

    except ResourceNotFoundError:
        raise FileNotFoundError(
            f"Blob '{blob_path}' was not found in container '{container_name}'"
        )

def download_pdf_from_blob(blob_path: str, container_name: str) -> str:
    """Download a PDF from Azure Blob Storage into a local temporary file."""
    return download_blob_to_temp_file(blob_path=blob_path, container_name=container_name)


def parse_blob_subject(subject: str):
    """
    Parses a blob subject string with the structure:
    /blobServices/default/containers/{container_name}/blobs/{blob_path}

    Returns:
        tuple: (blob_path: str, container_name: str)
    """
    parts = subject.strip("/").split("/")

    try:
        container_index = parts.index("containers") + 1
        blobs_index = parts.index("blobs") + 1
    except ValueError:
        raise ValueError("Invalid subject format: missing 'containers' or 'blobs' segment")

    container_name = parts[container_index]
    blob_parts = parts[blobs_index:]

    if not blob_parts:
        raise ValueError("Invalid subject format: blob path is missing")

    blob_path = "/".join(blob_parts)
    return blob_path, container_name
