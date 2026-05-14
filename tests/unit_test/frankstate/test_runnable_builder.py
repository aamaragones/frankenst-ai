import asyncio

import pytest

from tests.support.frankstate_doubles.builders import (
    FakeRunnableBuilder,
    FakeVectorStore,
)


@pytest.mark.unit
def test_runnable_builder_get_caches_configured_runnable() -> None:
    builder = FakeRunnableBuilder(sync_result="sync-result")

    first = builder.get()
    second = builder.get()

    assert first is second
    assert builder.configure_calls == 1


@pytest.mark.unit
def test_runnable_builder_invoke_and_ainvoke_delegate_to_configured_runnable() -> None:
    builder = FakeRunnableBuilder(sync_result="sync-result", async_result="async-result")

    assert builder.invoke("payload") == "sync-result"
    assert asyncio.run(builder.ainvoke("payload")) == "async-result"
    assert builder.configure_calls == 1
    assert builder.get().calls == [("invoke", "payload"), ("ainvoke", "payload")]


@pytest.mark.unit
def test_runnable_builder_prefers_explicit_retriever() -> None:
    retriever = object()
    builder = FakeRunnableBuilder(retriever=retriever)

    assert builder.get_retriever() is retriever


@pytest.mark.unit
def test_runnable_builder_get_retriever_builds_from_vectordb_once() -> None:
    retriever = object()
    vectordb = FakeVectorStore(retriever=retriever)
    builder = FakeRunnableBuilder(vectordb=vectordb)

    first = builder.get_retriever()
    second = builder.get_retriever()

    assert first is retriever
    assert second is retriever
    assert vectordb.calls == [{}]


@pytest.mark.unit
def test_runnable_builder_build_retriever_passes_kwargs_to_vectordb() -> None:
    retriever = object()
    vectordb = FakeVectorStore(retriever=retriever)
    builder = FakeRunnableBuilder(vectordb=vectordb)

    built = builder._build_retriever(search_type="mmr")

    assert built is retriever
    assert vectordb.calls == [{"search_type": "mmr"}]


@pytest.mark.unit
def test_runnable_builder_requires_retriever_source() -> None:
    builder = FakeRunnableBuilder(retriever=None, vectordb=None)

    with pytest.raises(ValueError, match="cannot build a retriever"):
        builder._build_retriever()