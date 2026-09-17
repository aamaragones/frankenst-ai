from collections.abc import Callable
from typing import cast

import pytest
from langchain_core.language_models import BaseChatModel

import main as main_module
from services.llm.llm_services import LLMRuntime
from tests.support.core_ai_examples_doubles import ToolBindingFakeModel

pytestmark = pytest.mark.unit


def test_main_prints_the_mermaid_of_the_chosen_layout(
    published_runtime: Callable[..., LLMRuntime], capsys: pytest.CaptureFixture[str]
) -> None:
    published_runtime(model=cast(BaseChatModel, ToolBindingFakeModel()))

    assert main_module.main(["--layout", "simple_oak"]) == 0
    out = capsys.readouterr().out
    assert "OakLangAgent" in out and "OakTools" in out


def test_main_rejects_an_unknown_layout() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main_module.main(["--layout", "nope"])
    assert excinfo.value.code == 2
