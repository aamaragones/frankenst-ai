from typing import Any

from frankstate.entity.runnable_builder import RetrieverMixin, RunnableBuilder


class SpyRunnable:
    def __init__(self, sync_result: Any = None, async_result: Any = None):
        self.sync_result = sync_result
        self.async_result = async_result if async_result is not None else sync_result
        self.calls: list[tuple[str, Any]] = []

    def _resolve_result(self, result: Any, payload: Any) -> Any:
        return result(payload) if callable(result) else result

    def invoke(self, payload: Any) -> Any:
        self.calls.append(("invoke", payload))
        return self._resolve_result(self.sync_result, payload)

    async def ainvoke(self, payload: Any) -> Any:
        self.calls.append(("ainvoke", payload))
        return self._resolve_result(self.async_result, payload)


class FakeVectorStore:
    def __init__(self, retriever: Any = None):
        self.retriever = retriever if retriever is not None else object()
        self.calls: list[dict[str, Any]] = []

    def as_retriever(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.retriever


class FakeRunnableBuilder(RetrieverMixin, RunnableBuilder):
    def __init__(
        self,
        sync_result: Any = "sync-result",
        async_result: Any = "async-result",
        retriever: Any = None,
        vectordb: Any = None,
    ):
        self.sync_result = sync_result
        self.async_result = async_result
        self.configure_calls = 0
        super().__init__(
            model=object(),
            retriever=retriever,
            vectordb=vectordb,
        )

    def _configure_runnable(self) -> SpyRunnable:
        self.configure_calls += 1
        return SpyRunnable(sync_result=self.sync_result, async_result=self.async_result)