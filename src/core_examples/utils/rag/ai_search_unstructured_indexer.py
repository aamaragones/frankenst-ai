import base64
import datetime as dt
import io
import os
import uuid
from typing import Any

from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from azure.search.documents.models import IndexingResult
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from unstructured.partition.pdf import partition_pdf

from core_examples.utils.rag.ai_search_schemas import (
    load_registered_ai_search_index_definition,
)


class AISearchMultiVectorDocumentIndexer:
    def __init__(
        self,
        search_client: SearchClient,
        llm_multimodal: BaseLanguageModel | None = None,
        embeddings: Embeddings | None = None,
    ):
        """
        Processes and manages documents for Azure AI Search with multi-vector indexing.

        Supports parsing PDFs (local or cloud), chunking content (text, tables, images), summarizing with a multimodal LLM,
        embedding with vector models, and indexing in Azure Search. Also allows manual upload and deletion of documents.

        Maintains internal state for elements, summaries, and chunks for retrieval or debugging.

        Args:
            search_client (SearchClient, optional): Azure AI Search client used to upload in a index the processed documents.
            llm_multimodal (BaseLanguageModel): Multimodal model for summarizing text, tables and images.
            embeddings (Embeddings, optional): An embedding model used to generate vectors.
        """
        self.search_client = search_client
        self.llm_multimodal = llm_multimodal
        self.embeddings = embeddings
        
        # Internal state
        self.file_path: str | None = None
        self.elements: dict[str, list] = {"texts": [], "tables": [], "images": []}
        self.summaries: dict[str, list] = {"texts": [], "tables": [], "images": []}
        self.documents: list[dict[str, Any]] = []


    def load_pdf(self, path: str) -> None:
        """
        Loads a PDF file from a local or temp path to split, embed and store.

        Args:
            path (str): Local or temp file path.

        Raises:
            FileNotFoundError: File not found.
        """
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        elif path:
            self.file_path = path

    def split_pdf(self, min_image_size: tuple[int, int] | None = None):
        """
        Splits the loaded PDF into texts, tables, and base64-encoded images.
        Updates internal state.

        Args:
            min_image_size (tuple[int, int], optional): Minimum `(width, height)` for extracted images.
                Images smaller than this threshold are ignored.

        Returns:
            tuple: Lists of texts, tables, and images.
        """
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

        texts, tables, images_b64 = [], [], []

        for chunk in chunks:            
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            
            elif "CompositeElement" in str(type(chunk)):
                has_table = False
                for el in chunk.metadata.orig_elements:
                    if "Table" in str(type(el)):
                        has_table = True
                    if "Image" in str(type(el)):
                        image_base64 = el.metadata.image_base64
                        if self._should_keep_image(image_base64, min_image_size):
                            images_b64.append(image_base64)
                
                if has_table:
                    tables.append(chunk)
                else:
                    texts.append(chunk)

        self.elements = {
            "texts": texts, 
            "tables": tables, 
            "images": images_b64
            }
        
        return texts, tables, images_b64

    def _should_keep_image(
        self,
        image_base64: str,
        min_image_size: tuple[int, int] | None,
    ) -> bool:
        """Return whether an extracted image should be kept for downstream indexing."""

        if min_image_size is None:
            return True

        try:
            width, height = self._get_image_size(image_base64)
        except Exception:
            return True

        min_width, min_height = min_image_size
        return width >= min_width and height >= min_height

    def _get_image_size(self, image_base64: str) -> tuple[int, int]:
        """Decode a base64 image payload and return its `(width, height)` dimensions."""

        image_bytes = base64.b64decode(image_base64)
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.size

    def _get_text_table_summary_chain(self):
        prompt_text = """
        You are an assistant in charge of summarizing tables and text.

        Provide a concise summary of the table or text.

        Reply with only the summary, without additional comments.
        Do not begin your message by saying "Here's a summary" or something similar.
        Simply provide the summary as is.

        Table or text fragment: {element}
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        return {"element": lambda x: x} | prompt | self.llm_multimodal | StrOutputParser()

    def _get_image_summary_chain(self):
        prompt_text = """
        Describe the image in detail. 
        """

        messages = [
            (
                "user",
                [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt | self.llm_multimodal | StrOutputParser()

    @retry(
        wait=wait_exponential(multiplier=1, min=4, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(HttpResponseError),
        reraise=True
    )
    def _retry_batch(
        self,
        chain: RunnableSequence,
        inputs: list[Any],
        config: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Executes a `.batch()` call with retry on HTTP 429 or transient errors.

        Args:
            chain (RunnableSequence): Runnable chain to execute.
            inputs (list): Inputs to pass in batch.
            config (dict, optional): Execution config (e.g., concurrency).

        Returns:
            list: Results from batch execution.
        """
        return chain.batch(inputs, config or {"max_concurrency": 3})

    def summarize_elements(self):
        """
        Summarizes the elements (texts, tables, images) extracted from the PDF.
        Updates internal state.

        Returns:
            tuple: Lists of summaries for texts, tables, and images.
        """
        summarize_chain = self._get_text_table_summary_chain()
        image_chain = self._get_image_summary_chain()

        formatted_table_html = [t.text + t.metadata.text_as_html for t in self.elements["tables"]] # ad to the table formatted to the text
        formatted_imgs = [{"image": b64} for b64 in self.elements["images"]]

        text_summaries = self._retry_batch(summarize_chain, self.elements["texts"]) # text already obtain as str the text
        table_summaries = self._retry_batch(summarize_chain, formatted_table_html)
        image_summaries = self._retry_batch(image_chain, formatted_imgs, config={})

        self.summaries = {
            "texts": text_summaries,
            "tables": table_summaries,
            "images": image_summaries
        }

        return text_summaries, table_summaries, image_summaries

    def embed_ai_search_index_documents(self) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        
        for content_type in ["texts", "tables", "images"]:
            chunks_list = self.elements.get(content_type)
            summaries_list = self.summaries.get(content_type)
            embeddings_summaries_list = self.embeddings.embed_documents(summaries_list)

            if chunks_list and summaries_list and len(chunks_list) == len(summaries_list):
                for i, chunk in enumerate(chunks_list):
                    doc_id = str(uuid.uuid4())
                    doc_type = content_type
                    summary = summaries_list[i]
                    embeddings_summary = embeddings_summaries_list[i]
                    metadata: dict[str, Any] = {}

                    # Metadatos opcionales
                    try:
                        chunk_metadata = getattr(chunk, "metadata", None)
                        languages = getattr(chunk_metadata, "languages", None) or ["und"]
                        metadata = {
                            "languages": ",".join(languages),
                            "last_modified": dt.datetime.now(dt.UTC),
                            "page_number": getattr(chunk_metadata, "page_number", 1),
                            "file_directory": getattr(chunk_metadata, "file_directory", None),
                            "filename": os.path.basename(self.file_path),
                            "filetype": getattr(chunk_metadata, "filetype", None),
                            "uri": f"https://www.devops.wiki/{getattr(chunk_metadata, 'filename', None)}" # TODO: improve dinamic url
                        }
                        # Eliminar claves con valor None
                        metadata = {k: v for k, v in metadata.items() if v is not None}
                    except Exception:
                        pass

                    document = {
                        "id": doc_id,
                        "type": doc_type,
                        "summary": summary,
                        "content": str(chunk),  # puede ser string (texto/base64) o estructura table
                        "metadata": metadata,
                        "embeddings": embeddings_summary,
                    }
                    documents.append(document)

        self.documents = documents

        return documents
    
    def upload_documents(self, documents: list[dict[str, Any]] | None = None) -> list[IndexingResult]:
        if documents:
            return self.search_client.upload_documents(documents=documents)
        else:
            return self.search_client.upload_documents(documents=self.documents)
    
    def delete_document_by_filename(self, filename: str, filter: str | None = None):
        if not filter:

            filter = (
                f"metadata/filename eq '{filename}'" # TODO: filter and search filename in other index...s
            )

        results = self.search_client.search(
            search_text="*",  
            filter=filter,
            select=["id"]
        )

        doc_ids_to_delete = [{"id": doc["id"]} for doc in results]

        if doc_ids_to_delete:
            self.search_client.delete_documents(documents=doc_ids_to_delete)
        else:
            raise ValueError(f"No documents found with filename: {filename}")


class AISearchIndexManager:
    def __init__(self, index_client: SearchIndexClient, index_name: str):
        """
        Manages the lifecycle and configuration of an Azure AI Search index, 
        including creation, updating, deletion, and retrieval of the index definition.

        Args:
            index_client (SearchIndexClient): The Azure SearchIndexClient to interact with the service.
            index_name (str): The name of the index to manage.
        """
        self.index_client = index_client
        self.index_name = index_name
    
    def get_index(self):
        """
        Retrieves the definition of the current index.

        Returns:
            SearchIndex: The current index definition.
        """
        return self.index_client.get_index(name=self.index_name)
    
    def index_exists(self) -> bool:
        """
        Checks whether the specified search index exists in the Azure Cognitive Search service.

        Returns:
            bool: True if the index exists, False otherwise.
        """
        try:
            self.index_client.get_index(name=self.index_name)
            return True
        except Exception:
            return False
    
    def create_index(self, registered_index_name: str | None = None):
        """
        Creates a new search index from a registered Azure AI Search name if it does not already exist.

        Args:
            registered_index_name (str | None): Registered Azure AI Search index
                name used to resolve the shared schema. Defaults to
                `self.index_name`.

        Raises:
            RuntimeError: If the index already exists.
        """
        if not self.index_exists():
            index = self._load_index_definition(registered_index_name)
            self.index_client.create_index(index)
        else:
            raise RuntimeError(f"The index '{self.index_name}' already exists. Please update it instead.")

    def update_index(self, registered_index_name: str | None = None):
        """
        Updates an existing index from a registered Azure AI Search name.

        Args:
            registered_index_name (str | None): Registered Azure AI Search index
                name used to resolve the shared schema. Defaults to
                `self.index_name`.
        """
        index = self._load_index_definition(registered_index_name)
        self.index_client.create_or_update_index(index=index)

    def delete_index(self):
        """
        Deletes the current search index.
        """
        self.index_client.delete_index(self.index_name)

    def _load_index_definition(self, registered_index_name: str | None = None) -> SearchIndex:
        """Load a registered index definition and bind it to `self.index_name`."""

        return load_registered_ai_search_index_definition(
            index_name=registered_index_name or self.index_name,
            runtime_index_name=self.index_name,
        )