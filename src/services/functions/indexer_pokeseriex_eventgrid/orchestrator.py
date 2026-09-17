from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient

from config.settings import get_settings
from services.llm.llm_services import LLMServices
from utils.blob_storage import download_pdf_from_blob, parse_blob_subject
from utils.rag.ai_search_unstructured_indexer import (
    AISearchIndexManager,
    AISearchMultiVectorDocumentIndexer,
)

INDEX_NAME = "pokeseriex-index"


class Orchestrator:
    """Coordinate Azure EventGrid indexing flows for the repository example.

    The trigger handler may inject pre-built Azure Search clients so a single
    invocation does not repeat Key Vault lookups for every orchestrator step.
    """

    @staticmethod
    def create_search_clients(
        index_name: str,
    ) -> tuple[SearchIndexClient, SearchClient]:
        """Create the Azure Search clients used by the EventGrid flow."""
        settings = get_settings()
        service_endpoint = settings.resolve_secret("AZURE_SEARCH_SERVICE_ENDPOINT")
        key = settings.resolve_secret("AZURE_SEARCH_API_KEY")
        credential = AzureKeyCredential(key)
        return (
            SearchIndexClient(service_endpoint, credential),
            SearchClient(service_endpoint, index_name, credential),
        )

    @staticmethod
    def check_index(
        index_name: str, index_client: SearchIndexClient | None = None
    ) -> None:
        if index_client is None:
            index_client, _ = Orchestrator.create_search_clients(index_name)

        # Initialize ai_search_index_manager
        ai_search_index_manager = AISearchIndexManager(
            index_name=index_name, index_client=index_client
        )

        if not ai_search_index_manager.index_exists():
            # Create index
            ai_search_index_manager.create_index()

    @staticmethod
    def get_index_name() -> str:
        return INDEX_NAME

    @staticmethod
    def document_indexing(
        index_name: str, subject: str, search_client: SearchClient | None = None
    ) -> None:
        if search_client is None:
            _, search_client = Orchestrator.create_search_clients(index_name)

        # Parse the subject
        blob_path, container_name = parse_blob_subject(subject=subject)
        filename = blob_path.split("/")[-1]

        # Load file from blob
        temp_filepath = download_pdf_from_blob(
            blob_path=blob_path, container_name=container_name
        )

        model, embeddings = LLMServices.launch().require("model", "embeddings")
        indexer = AISearchMultiVectorDocumentIndexer(search_client, model, embeddings)

        if filename.endswith(".pdf"):
            indexer.load_pdf(temp_filepath)
            indexer.split_pdf()
            indexer.summarize_elements()
            indexer.embed_ai_search_index_documents()
            indexer.upload_documents()
        else:
            raise ValueError("Provide a pdf file")

    @staticmethod
    def delete_document_by_filename(
        index_name: str, subject: str, search_client: SearchClient | None = None
    ) -> None:
        if search_client is None:
            _, search_client = Orchestrator.create_search_clients(index_name)

        # Parse the subject
        blob_path, _ = parse_blob_subject(subject=subject)
        filename = blob_path.split("/")[-1]

        # Initialize ai_search_index_manager
        indexer = AISearchMultiVectorDocumentIndexer(search_client)
        indexer.delete_document_by_filename(filename=filename)
