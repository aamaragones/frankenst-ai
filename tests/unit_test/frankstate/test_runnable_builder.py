import asyncio

import pytest

from tests.support.frankstate_doubles.builders import (
    FakeRunnableBuilder,
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