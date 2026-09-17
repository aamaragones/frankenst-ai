from typing import cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel

from core_ai_examples.components.runnables.structured_grade_document.structured_grade_document import (
    StructuredGradeDocument,
)
from core_ai_examples.models.structured_output.grade_documents import GradeDocuments
from tests.support.core_ai_examples_doubles import (
    FakeResponsesStructuredModel,
    FakeStructuredModel,
)

pytestmark = pytest.mark.unit

PAYLOAD = {
    "question": "Who caught Caterpie?",
    "context": {"texts": ["Ash caught Caterpie."], "images": []},
}


def _build(
    model: FakeStructuredModel | FakeResponsesStructuredModel,
) -> StructuredGradeDocument:
    return StructuredGradeDocument(
        model=cast(BaseChatModel, model), structured_output_schema=GradeDocuments
    )


def test_build_prompt_requires_question_and_context() -> None:
    with pytest.raises(ValueError, match="Missing required keys"):
        _build(FakeStructuredModel())._build_prompt(question="q")


def test_chat_completions_route_uses_with_structured_output() -> None:
    model = FakeStructuredModel()

    result = _build(model).invoke(PAYLOAD)

    assert isinstance(result, GradeDocuments) and result.binary_score == "yes"
    assert model.structured_schema is GradeDocuments
    assert model.structured_method == "json_schema"
    rendered = model.captured_prompts[0].to_messages()[0].content[0]["text"]
    assert "Ash caught Caterpie." in rendered and "Who caught Caterpie?" in rendered


def test_responses_route_binds_the_schema_under_text_format() -> None:
    model = FakeResponsesStructuredModel()

    result = _build(model).invoke(PAYLOAD)

    assert isinstance(result, GradeDocuments) and result.binary_score == "yes"
    assert "response_format" not in model.bound_kwargs
    schema = model.bound_kwargs["text"]["format"]
    assert schema["type"] == "json_schema"
    assert schema["name"] == "GradeDocuments"
    assert schema["strict"] is True
    assert schema["schema"]["required"] == ["binary_score"]
