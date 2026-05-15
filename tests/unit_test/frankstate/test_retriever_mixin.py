import pytest

from tests.support.frankstate_doubles.builders import (
    FakeRunnableBuilder,
    FakeVectorStore,
)


@pytest.mark.unit
def test_retriever_mixin_prefers_explicit_retriever() -> None:
    retriever = object()
    builder = FakeRunnableBuilder(retriever=retriever)

    assert builder.built_retriever is retriever


@pytest.mark.unit
def test_retriever_mixin_builds_from_vectordb_once() -> None:
    retriever = object()
    vectordb = FakeVectorStore(retriever=retriever)
    builder = FakeRunnableBuilder(vectordb=vectordb)

    first = builder.built_retriever
    second = builder.built_retriever

    assert first is retriever
    assert second is retriever
    assert vectordb.calls == [{}]


@pytest.mark.unit
def test_retriever_mixin_build_retriever_passes_kwargs_to_vectordb() -> None:
    retriever = object()
    vectordb = FakeVectorStore(retriever=retriever)
    builder = FakeRunnableBuilder(vectordb=vectordb)

    built = builder._build_retriever(search_type="mmr")

    assert built is retriever
    assert vectordb.calls == [{"search_type": "mmr"}]


@pytest.mark.unit
def test_retriever_mixin_requires_retriever_source() -> None:
    builder = FakeRunnableBuilder(retriever=None, vectordb=None)

    with pytest.raises(ValueError, match="cannot build a retriever"):
        builder._build_retriever()
