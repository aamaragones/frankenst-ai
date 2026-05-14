from typing import Any

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langgraph.graph import END, START

from core_examples.components.edges.evaluators.grade_rewrite_generate import (
    GradeRewriteGenerate,
)
from core_examples.components.nodes.enhancers.generate_answer_ainvoke import (
    GenerateAnswerAsyncInvoke,
)
from core_examples.components.nodes.enhancers.retrieve_context_ai_search import (
    RetrieveContextAISearch,
)
from core_examples.components.nodes.enhancers.rewrite_question_ainvoke import (
    RewriteQuestionAsyncInvoke,
)
from core_examples.components.retrievers.ai_search_multivector_retriever.ai_search_multivector_retriever import (
    AISearchMultiVectorRetriever,
)
from core_examples.components.runnables.multimodal_generation.multimodal_generation import (
    MultimodalGeneration,
)
from core_examples.components.runnables.rewrite_question.rewrite_question import (
    RewriteQuestion,
)
from core_examples.components.runnables.structured_grade_document.structured_grade_document import (
    StructuredGradeDocument,
)
from core_examples.config.settings import get_settings
from core_examples.models.structured_output.grade_documents import GradeDocuments
from core_examples.utils.config_loader import load_node_registry
from core_examples.utils.key_vault import get_secret
from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.entity.graph_layout import GraphLayout
from frankstate.entity.node import SimpleNode
from services.foundry.llms import LLMServices


# NOTE: This is an example implementation for illustration purposes
# NOTE: Here you can add other subgraphs as nodes
class AISearchAdaptiveRAGConfigGraph(GraphLayout):
    """Adaptive RAG layout backed by Azure AI Search.

    State expectations:
        - Uses `RAGState` or a compatible schema with `messages`, `question`,
          `context`, `generation` and `iterations`.

    Flow:
        START -> Retriever -> (Generation | Rewrite)
        Rewrite -> Retriever
        Generation -> END

    Use this layout when retrieval is delegated to Azure AI Search instead of a
    local vector store.
    """

    CONFIG_NODES: dict[str, Any]
    GENERARION_CHAIN: MultimodalGeneration
    GRADE_STRUCTURED_CHAIN: StructuredGradeDocument
    REWRITE_CHAIN: RewriteQuestion
    RAW_RETRIEVER: AISearchMultiVectorRetriever

    INDEX_NAME = "demo-rag-multimodal-index"

    def build_runtime(self) -> dict[str, Any]:
        settings = get_settings()
        LLMServices.launch()
        if LLMServices.model is None or LLMServices.embeddings is None:
            raise RuntimeError("LLMServices.launch() did not initialize model and embeddings.")

        service_endpoint = get_secret("AZURE_SEARCH_SERVICE_ENDPOINT")
        key = get_secret("AZURE_SEARCH_API_KEY")
        search_client = SearchClient(
            service_endpoint,
            self.INDEX_NAME,
            AzureKeyCredential(key),
        )
        raw_retriever = AISearchMultiVectorRetriever(
            embeddings=LLMServices.embeddings,
            search_client=search_client,
        )

        return {
            "CONFIG_NODES": load_node_registry(settings.config_nodes_file_path),
            "RAW_RETRIEVER": raw_retriever,
            "GENERARION_CHAIN": MultimodalGeneration(model=LLMServices.model),
            "GRADE_STRUCTURED_CHAIN": StructuredGradeDocument(
                model=LLMServices.model,
                structured_output_schema=GradeDocuments,
            ),
            "REWRITE_CHAIN": RewriteQuestion(model=LLMServices.model),
        }

    def layout(self) -> None:
        ## NODES
        self.GENERATION_NODE = SimpleNode(
            enhancer=GenerateAnswerAsyncInvoke(self.GENERARION_CHAIN),
            name=self.CONFIG_NODES["GENERATION_NODE"]["name"],
            tags=[self.CONFIG_NODES["GENERATION_NODE"]["description"]],
        )
        self.RETRIEVER_NODE = SimpleNode(
            enhancer=RetrieveContextAISearch(retriever=self.RAW_RETRIEVER),
            name=self.CONFIG_NODES["RETRIEVER_NODE"]["name"],
            tags=[self.CONFIG_NODES["RETRIEVER_NODE"]["description"]],
        )
        self.REWRITE_NODE = SimpleNode(
            enhancer=RewriteQuestionAsyncInvoke(self.REWRITE_CHAIN),
            name=self.CONFIG_NODES["REWRITE_NODE"]["name"],
            tags=[self.CONFIG_NODES["REWRITE_NODE"]["description"]],
        )

        ## EDGES
        self._EDGE_1 = SimpleEdge(node_source=START, node_path=self.RETRIEVER_NODE.name)
        self._EDGE_2 = ConditionalEdge(
            evaluator=GradeRewriteGenerate(self.GRADE_STRUCTURED_CHAIN),
            map_dict={
                "generate": self.GENERATION_NODE.name,
                "rewrite": self.REWRITE_NODE.name,
            },
            node_source=self.RETRIEVER_NODE.name,
        )
        self._EDGE_3 = SimpleEdge(node_source=self.GENERATION_NODE.name, node_path=END)
        self._EDGE_4 = SimpleEdge(
            node_source=self.REWRITE_NODE.name,
            node_path=self.RETRIEVER_NODE.name,
        )
