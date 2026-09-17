import logging
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel

from frankstate.entity.runnable_builder import PromptMixin, RunnableBuilder
from utils.common import load_and_clean_text_file, resolve_package_resource


class StructuredGradeDocument(PromptMixin, RunnableBuilder):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, model: BaseChatModel, structured_output_schema: type[BaseModel]):
        super().__init__(model=model)
        self.structured_output_schema = structured_output_schema
        self.logger.info("StructuredGradeDocument initialized")

    @staticmethod
    def _fragment(name: str) -> str:
        package = __package__ or __name__
        return load_and_clean_text_file(
            resolve_package_resource(package, "prompt", f"{name}.md")
        )

    def _build_prompt(self, **kwargs: Any) -> ChatPromptTemplate:
        docs_by_type = kwargs.get("context")
        question = kwargs.get("question")
        if not question or not isinstance(docs_by_type, dict):
            raise ValueError("Missing required keys 'question' and 'context' in kwargs")

        prompt_template = self._fragment("format_template").format(
            context=self._fragment("context"),
            retrieved_context=docs_by_type["texts"],
            question=question,
            instructions=self._fragment("instructions"),
        )
        prompt_content: list[str | dict[str, Any]] = [
            {"type": "text", "text": prompt_template}
        ]
        prompt_content.extend(docs_by_type["images"])
        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    def _structured_model(self) -> Runnable[Any, Any]:
        # The Responses route rejects `response_format`; its equivalent is `text.format`.
        if getattr(self.model, "use_responses_api", False):
            schema = convert_to_openai_function(
                self.structured_output_schema, strict=True
            )
            schema["schema"] = schema.pop("parameters")
            return self.model.bind(
                text={"format": {"type": "json_schema", **schema}}
            ) | PydanticOutputParser(pydantic_object=self.structured_output_schema)
        return self.model.with_structured_output(
            schema=self.structured_output_schema, method="json_schema"
        )

    def _configure_runnable(self) -> Runnable[Any, Any]:
        return (
            {
                "context": RunnableLambda(
                    lambda kwargs: cast(dict[str, Any], kwargs)["context"]
                ),
                "question": RunnableLambda(
                    lambda kwargs: cast(dict[str, Any], kwargs)["question"]
                ),
            }
            | RunnableLambda(lambda kwargs: self._build_prompt(**kwargs))
            | self._structured_model()
        )
