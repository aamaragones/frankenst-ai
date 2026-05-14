import base64
import binascii
import io
from typing import Any

from langchain_core.documents import Document
from PIL import Image


def _extract_base64_image_payload(doc: Any) -> str | None:
    """Return a normalized base64 payload only when the input is a valid image."""

    if isinstance(doc, bytes):
        try:
            candidate = doc.decode("ascii")
        except UnicodeDecodeError:
            return None
    elif isinstance(doc, str):
        candidate = doc
    else:
        return None

    payload = candidate.split(",", 1)[1] if candidate.startswith("data:image") and "," in candidate else candidate

    try:
        image_data = base64.b64decode(payload, validate=True)
        with Image.open(io.BytesIO(image_data)) as image:
            image.verify()
    except (binascii.Error, ValueError, OSError):
        return None

    return payload

def parse_docs(docs: list[Any]) -> dict[str, list[Any]]:
    """Split retrieved multimodal documents into image payloads and text payloads."""

    b64 = []
    text = []
    for doc in docs:
        if isinstance(doc, Document):
            content_type = doc.metadata.get("content_type")
            if content_type == "images":
                b64.append(doc.page_content)
            else:
                text.append(doc)
            continue

        image_payload = _extract_base64_image_payload(doc)
        if image_payload is not None:
            b64.append(image_payload)
        else:
            text.append(doc)
    return {"images": b64, "texts": text}

def parse_context(docs_by_type: dict[str, list[Any]]) -> dict[str, Any]:
    """Build multimodal context from retrieved text and image payloads."""

    context_text = ""
    context_images = []

    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            if isinstance(text_element, Document):
                context_text += text_element.page_content + "\n"
            elif hasattr(text_element, "text"):
                context_text += text_element.text + "\n"
            else:
                context_text += str(text_element) + "\n"

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            context_images.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image}"}
            })

    return {
        "texts": context_text.strip(),
        "images": context_images
    }

def show_base64_image(base64_string: str) -> dict:
    # Remove any header like "data:image/png;base64," if present
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    image.show()