"""Adaptive RAG layout backed by the local Chroma store."""

from typing import Any

from langgraph.graph import END, START

from config.settings import get_settings
from core_ai_examples.components.edges.evaluators.grade_rewrite_generate import (
    GradeRewriteGenerate,
)
from core_ai_examples.components.nodes.enhancers.generate_answer_ainvoke import (
    GenerateAnswerAsyncInvoke,
)
from core_ai_examples.components.nodes.enhancers.retrieve_context_ainvoke import (
    RetrieveContextAsyncInvoke,
)
from core_ai_examples.components.nodes.enhancers.rewrite_question_ainvoke import (
    RewriteQuestionAsyncInvoke,
)
from core_ai_examples.components.retrievers.langchain_chroma_multivector_retriever.langchain_chroma_multivector_retriever import (
    LangchainChromaMultiVectorRetriever,
)
from core_ai_examples.components.runnables.multimodal_generation.multimodal_generation import (
    MultimodalGeneration,
)
from core_ai_examples.components.runnables.multimodal_retriever.multimodal_retriever import (
    MultimodalRetriever,
)
from core_ai_examples.components.runnables.rewrite_question.rewrite_question import (
    RewriteQuestion,
)
from core_ai_examples.components.runnables.structured_grade_document.structured_grade_document import (
    StructuredGradeDocument,
)
from core_ai_examples.models.structured_output.grade_documents import GradeDocuments
from frankstate.entity.edge import ConditionalEdge, SimpleEdge
from frankstate.entity.graph_layout import GraphLayout
from frankstate.entity.node import SimpleNode
from services.llm.llm_services import LLMServices
from utils.config_loader import load_node_registry


# NOTE: This is an example implementation for illustration purposes
# NOTE: Here you can add other subgraphs as nodes
class LocalVectorStoreAdaptiveRAGConfigGraph(GraphLayout):
    """Adaptive RAG layout backed by a local vector store retriever.

    State expectations:
        - Uses `RAGState` or a compatible schema with `messages`, `question`,
          `context`, `generation` and `iterations`.

    Flow:
        START -> Retriever -> (Generation | Rewrite)
        Rewrite -> Retriever
        Generation -> END

    This layout is the reference pattern for local multimodal retrieval loops.
    """

    CONFIG_NODES: dict[str, Any]
    RETRIEVER_CHAIN: MultimodalRetriever
    GENERATION_CHAIN: MultimodalGeneration
    GRADE_STRUCTURED_CHAIN: StructuredGradeDocument
    REWRITE_CHAIN: RewriteQuestion

    def build_runtime(self) -> dict[str, Any]:
        settings = get_settings()
        model, embeddings = LLMServices.launch().require("model", "embeddings")

        raw_retriever = LangchainChromaMultiVectorRetriever(
            embeddings=embeddings,
        ).get_retriever()

        return {
            "CONFIG_NODES": load_node_registry(settings.config_nodes_file_path),
            "RETRIEVER_CHAIN": MultimodalRetriever(
                model=model,
                retriever=raw_retriever,
            ),
            "GENERATION_CHAIN": MultimodalGeneration(model=model),
            "GRADE_STRUCTURED_CHAIN": StructuredGradeDocument(
                model=model,
                structured_output_schema=GradeDocuments,
            ),
            "REWRITE_CHAIN": RewriteQuestion(model=model),
        }

    def layout(self) -> None:
        ## NODES
        self.GENERATION_NODE = SimpleNode(
            enhancer=GenerateAnswerAsyncInvoke(self.GENERATION_CHAIN),
            name=self.CONFIG_NODES["GENERATION_NODE"]["name"],
            metadata=self.CONFIG_NODES["GENERATION_NODE"]["metadata"],
        )
        self.RETRIEVER_NODE = SimpleNode(
            enhancer=RetrieveContextAsyncInvoke(self.RETRIEVER_CHAIN),
            name=self.CONFIG_NODES["RETRIEVER_NODE"]["name"],
            metadata=self.CONFIG_NODES["RETRIEVER_NODE"]["metadata"],
        )
        self.REWRITE_NODE = SimpleNode(
            enhancer=RewriteQuestionAsyncInvoke(self.REWRITE_CHAIN),
            name=self.CONFIG_NODES["REWRITE_NODE"]["name"],
            metadata=self.CONFIG_NODES["REWRITE_NODE"]["metadata"],
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
