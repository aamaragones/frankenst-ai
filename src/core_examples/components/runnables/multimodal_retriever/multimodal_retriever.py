import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableLambda

from core_examples.utils.rag.processing import parse_context, parse_docs
from frankstate.entity.runnable_builder import RunnableBuilder


class MultimodalRetriever(RunnableBuilder):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, model: BaseChatModel, retriever: BaseRetriever) -> None:
        """Compose a multimodal retrieval runnable from a pre-built retriever."""

        super().__init__(model=model, retriever=retriever)

        self.logger.info("MultimodalRetriever initialized")

    def _configure_runnable(self) -> Runnable:
        """Compose retriever output parsing into the runnable returned by invoke or ainvoke."""

        if self.retriever is None:
            raise ValueError("MultimodalRetriever requires a pre-built retriever.")

        retriever = self.retriever
        multimodal_retriever_parse_chain = retriever | RunnableLambda(parse_docs) | RunnableLambda(parse_context)

        return multimodal_retriever_parse_chain