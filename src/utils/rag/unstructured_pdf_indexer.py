"""Shared PDF partitioning and summarization behind the two multi-vector indexers."""

import base64
import io
import os
from typing import Any, Literal

from azure.core.exceptions import HttpResponseError
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

Bucket = Literal["texts", "tables"]

_TEXT_TABLE_PROMPT = """
You are an assistant in charge of summarizing tables and text.

Provide a concise summary of the table or text.

Reply with only the summary, without additional comments.
Do not begin your message by saying "Here's a summary" or something similar.
Simply provide the summary as is.

Table or text fragment: {element}
"""

_IMAGE_PROMPT = [
    (
        "user",
        [
            {"type": "text", "text": "Describe the image in detail."},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,{image}"},
            },
        ],
    )
]


class UnstructuredPDFIndexerBase:
    """Load, split and summarize a PDF; subclasses decide the sink and the table shape."""

    def __init__(
        self,
        text_model: BaseLanguageModel[Any] | None,
        image_model: BaseLanguageModel[Any] | None,
    ) -> None:
        self.text_model = text_model
        self.image_model = image_model
        self.file_path: str | None = None
        self.elements: dict[str, list[Any]] = {"texts": [], "tables": [], "images": []}
        self.summaries: dict[str, list[Any]] = {"texts": [], "tables": [], "images": []}

    def load_pdf(self, path: str) -> None:
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if path:
            self.file_path = path

    def _classify_chunk(self, chunk: Any) -> Bucket | None:
        """Bucket for a partition chunk; `None` drops it."""
        if "Table" in str(type(chunk)):
            return "tables"
        if "CompositeElement" in str(type(chunk)):
            return "texts"
        return None

    def _format_table(self, chunk: Any) -> str:
        return str(chunk.metadata.text_as_html)

    def split_pdf(
        self, min_image_size: tuple[int, int] | None = None
    ) -> tuple[list[Any], list[Any], list[str]]:
        from unstructured.partition.pdf import partition_pdf

        chunks = partition_pdf(
            filename=self.file_path,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            max_characters=10000,
            combine_text_under_n_chars=2000,
            new_after_n_chars=6000,
        )
        texts: list[Any] = []
        tables: list[Any] = []
        images_b64: list[str] = []
        for chunk in chunks:
            bucket = self._classify_chunk(chunk)
            if bucket is None:
                continue
            (tables if bucket == "tables" else texts).append(chunk)
            for element in getattr(chunk.metadata, "orig_elements", None) or []:
                if "Image" in str(type(element)):
                    image_base64 = element.metadata.image_base64
                    if self._should_keep_image(image_base64, min_image_size):
                        images_b64.append(image_base64)
        self.elements = {"texts": texts, "tables": tables, "images": images_b64}
        return texts, tables, images_b64

    def _should_keep_image(
        self, image_base64: str, min_image_size: tuple[int, int] | None
    ) -> bool:
        if min_image_size is None:
            return True
        try:
            width, height = self._get_image_size(image_base64)
        except Exception:
            return True
        min_width, min_height = min_image_size
        return width >= min_width and height >= min_height

    @staticmethod
    def _get_image_size(image_base64: str) -> tuple[int, int]:
        with Image.open(io.BytesIO(base64.b64decode(image_base64))) as image:
            return image.size

    def _text_table_summary_chain(self) -> Runnable[Any, str]:
        if self.text_model is None:
            raise RuntimeError(
                "A text model is required to summarize texts and tables."
            )
        prompt = ChatPromptTemplate.from_template(_TEXT_TABLE_PROMPT)
        return {"element": lambda x: x} | prompt | self.text_model | StrOutputParser()

    def _image_summary_chain(self) -> Runnable[Any, str]:
        if self.image_model is None:
            raise RuntimeError("An image model is required to summarize images.")
        prompt = ChatPromptTemplate.from_messages(_IMAGE_PROMPT)
        return prompt | self.image_model | StrOutputParser()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(HttpResponseError),
        reraise=True,
    )
    def _retry_batch(
        self,
        chain: Runnable[Any, Any],
        inputs: list[Any],
        config: RunnableConfig | None = None,
    ) -> list[Any]:
        return chain.batch(inputs, config or RunnableConfig(max_concurrency=3))

    def summarize_elements(self) -> tuple[list[Any], list[Any], list[Any]]:
        summarize_chain = self._text_table_summary_chain()
        image_chain = self._image_summary_chain()
        tables = [self._format_table(t) for t in self.elements["tables"]]
        images = [{"image": b64} for b64 in self.elements["images"]]

        text_summaries = self._retry_batch(summarize_chain, self.elements["texts"])
        table_summaries = self._retry_batch(summarize_chain, tables)
        image_summaries = self._retry_batch(
            image_chain, images, config=RunnableConfig()
        )
        self.summaries = {
            "texts": text_summaries,
            "tables": table_summaries,
            "images": image_summaries,
        }
        return text_summaries, table_summaries, image_summaries
