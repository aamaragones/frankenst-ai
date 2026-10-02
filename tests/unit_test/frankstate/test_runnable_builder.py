import asyncio
from collections.abc import Coroutine
from typing import Any, cast

import pytest

from tests.support.frankstate_doubles.builders import (
    FakeRunnableBuilder,
    SpyRunnable,
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
    builder = FakeRunnableBuilder(
        sync_result="sync-result", async_result="async-result"
    )

    assert builder.invoke("payload") == "sync-result"
    assert (
        asyncio.run(cast(Coroutine[Any, Any, Any], builder.ainvoke("payload")))
        == "async-result"
    )
    assert builder.configure_calls == 1
    assert cast(SpyRunnable, builder.get()).calls == [
        ("invoke", "payload"),
        ("ainvoke", "payload"),
    ]


@pytest.mark.unit
def test_runnable_builder_forwards_kwargs_such_as_config_to_the_runnable() -> None:
    builder = FakeRunnableBuilder(
        sync_result="sync-result", async_result="async-result"
    )
    config = {"configurable": {"thread_id": "t-1"}}

    builder.invoke("payload", config=config)
    asyncio.run(
        cast(Coroutine[Any, Any, Any], builder.ainvoke("payload", config=config))
    )

    assert cast(SpyRunnable, builder.get()).kwargs == [
        {"config": config},
        {"config": config},
    ]
