from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel


class RunnableBuilder(ABC):
    """Base helper for creating reusable LangChain runnables.

    Subclasses define how prompts, retrievers or chains are assembled, while
    this base class handles lazy construction and exposes a small invoke API
    used by enhancers and evaluators across the project.
    """

    def __init__(
        self,
        model: BaseChatModel,
        vectordb: VectorStore | None = None,
        retriever: BaseRetriever | None = None,
        tools: list[BaseTool] | None = None,
        structured_output_schema: type[BaseModel] | None = None,
    ):
        self.model = model
        self.vectordb = vectordb
        self.retriever = retriever
        self.tools = tools
        self.structured_output_schema = structured_output_schema

        # Start the runnable
        self._runnable: Runnable | None = None
        self._retriever: BaseRetriever | None = None
    
    def _build_prompt(self) -> ChatPromptTemplate:
        """Build the prompt runnable chain and return it.

        Subclasses may override this when prompt construction is a distinct
        step inside `_configure_runnable()`.
        """

        raise NotImplementedError(f"{self.__class__.__name__} does not implement `_build_prompt`.")
    
    def _build_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Builds and returns a retriever from the vectorstore.

        Subclasses can override this for custom logic. If `self.retriever` is
        already provided, it takes precedence over `self.vectordb`.
        """

        if self.retriever is not None:
            return self.retriever
        if self.vectordb is None:
            raise ValueError(f"{self.__class__.__name__} cannot build a retriever because `vectordb` is None.")
        return self.vectordb.as_retriever(**kwargs)

    def _require_runnable(self) -> Runnable:
        """Return the configured runnable, initializing it lazily when needed."""

        runnable = self._runnable
        if runnable is None:
            runnable = self._configure_runnable()
            self._runnable = runnable

        return runnable

    @abstractmethod
    def _configure_runnable(self) -> Runnable:
        """Configure the main runnable and return it.

        The configured object may be a chain, a retriever or any LangChain
        runnable compatible with `invoke()` and `ainvoke()`.
        """
        pass
    
    @property
    def _get_runnable(self) -> Runnable:
        """Lazily initialize and return the configured runnable."""

        return self._require_runnable()
    
    @property
    def _get_retriever(self) -> BaseRetriever:
        """Lazily initialize and return the configured retriever."""

        if self._retriever is None:
            self._retriever = self._build_retriever()
        return self._retriever

    def invoke(self, input: Any) -> Any:
        """Invoke the configured runnable synchronously.

        Use this convenience method when callers want to treat the builder
        itself as the execution surface. Use `get()` when another object needs
        the runnable instance directly for composition.
        """
        runnable = self._require_runnable()
        return runnable.invoke(input)


    def ainvoke(self, input: Any) -> Awaitable[Any]:
        """Invoke the configured runnable asynchronously.

        Use this convenience method when callers want to treat the builder
        itself as the execution surface. Use `get()` when another object needs
        the runnable instance directly for composition.
        """
        runnable = self._require_runnable()
        return runnable.ainvoke(input)
    
    def get(self) -> Runnable:
        """Return the lazily configured runnable instance.

        This is the preferred accessor when another object, such as a state
        handler, needs the runnable itself rather than an immediate invocation.
        """
        return self._require_runnable()

    def get_retriever(self) -> BaseRetriever:
        """Return the lazily configured retriever instance."""
        return self._get_retriever
