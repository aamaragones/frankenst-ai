from abc import ABC, abstractmethod
from collections.abc import Awaitable
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStore


class RunnableBuilder(ABC):
    """Base lifecycle contract for assembling LangChain LCEL runnable.
    Combine with `PromptMixin` and/or `RetrieverMixin` to add capabilities:

        class ChatBuilder(PromptMixin, RunnableBuilder):
            def __init__(self, model: BaseChatModel) -> None:
                super().__init__(model=model)

        def _build_prompt(self, **kwargs: Any) -> ChatPromptTemplate:
            return ChatPromptTemplate.from_messages([
                ("system", "You are an agent. Use your tools."),
                MessagesPlaceholder("messages"),
            ])

        def _configure_runnable(self) -> Runnable:
            return self._build_prompt() | self.model.bind_tools(self.tools)

    Args:
        model: Chat model used to build the runnable chain.
    """

    def __init__(self, *, model: BaseChatModel) -> None:
        self.model = model
        self._runnable: Runnable | None = None

    @abstractmethod
    def _configure_runnable(self) -> Runnable:
        """Build and return the runnable chain.
        
        Returns:
            A fully configured LangChain `Runnable` ready for `invoke` or
            `ainvoke`.
        """
        raise NotImplementedError

    def _require_runnable(self) -> Runnable:
        if self._runnable is None:
            self._runnable = self._configure_runnable()
        return self._runnable

    @property
    def runnable(self) -> Runnable:
        """The lazily initialized, cached runnable instance."""
        return self._require_runnable()

    def invoke(self, input: Any) -> Any:
        """Invoke the runnable synchronously."""
        return self.runnable.invoke(input)

    def ainvoke(self, input: Any) -> Awaitable[Any]:
        """Invoke the runnable asynchronously."""
        return self.runnable.ainvoke(input)

    def get(self) -> Runnable:
        """Return the runnable, building it on first call."""
        return self.runnable


class RetrieverMixin:
    """Cooperative mixin that adds a lazily initialized retriever to a builder.

        class MyVectorDBRetriever(RetrieverMixin, RunnableBuilder):
            def __init__(self, model: BaseChatModel, vectordb: VectorStore) -> None:
                super().__init__(model=model, vectordb=vectordb)

            def _configure_runnable(self) -> Runnable:
                return (
                    {"context": self.retriever, "question": RunnablePassthrough()}
                    | RunnableLambda(lambda x: self._build_prompt())
                    | self.model
                )
    Args:
        retriever: Pre-built retriever. Takes priority over `vectordb` when
            both are provided.
        vectordb: Vector store used to create a retriever via
            `as_retriever(**kwargs)`.
    """

    def __init__(
        self,
        *,
        vectordb: VectorStore | None = None,
        retriever: BaseRetriever | None = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self._vectordb = vectordb
        self._provided_retriever = retriever
        self._retriever: BaseRetriever | None = None

    def _build_retriever(self, **kwargs: Any) -> BaseRetriever:
        """Build and return a retriever instance.

        Args:
            **kwargs: Optional keyword arguments forwarded to
                `vectordb.as_retriever`. Common options include
                `search_type="mmr"` or `search_kwargs={"k": 5}`.

        Returns:
            The resolved retriever instance.

        Raises:
            ValueError: If neither `retriever` nor `vectordb` was provided at
                construction time.
        """
        if self._provided_retriever is not None:
            return self._provided_retriever

        if self._vectordb is None:
            raise ValueError(
                f"{self.__class__.__name__} cannot build a retriever: "
                "provide `retriever` or `vectordb` at construction time."
            )

        return self._vectordb.as_retriever(**kwargs)

    @property
    def retriever(self) -> BaseRetriever:
        """
        Lazily initialized retriever instance.
        """
        if self._retriever is None:
            self._retriever = self._build_retriever()
        return self._retriever


class PromptMixin(ABC):
    """Abstract mixin that enforces a `_build_prompt` hook on a builder.

        class ChatBuilder(PromptMixin, RunnableBuilder):
            def __init__(self, model: BaseChatModel) -> None:
                super().__init__(model=model)

            def _build_prompt(self, **kwargs: Any) -> ChatPromptTemplate:
                return ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant."),
                    ("human", "{input}"),
                ])

            def _configure_runnable(self) -> Runnable:
                return self._build_prompt() | self.model
    """

    @abstractmethod
    def _build_prompt(self, **kwargs: Any) -> ChatPromptTemplate:
        """Build and return the prompt template for this builder.

        Args:
            **kwargs: Runtime values passed through the chain payload
            (e.g. `context`, `question`).

        Returns:
            The prompt template wired into the runnable chain by
            `_configure_runnable`.
        """
        raise NotImplementedError

