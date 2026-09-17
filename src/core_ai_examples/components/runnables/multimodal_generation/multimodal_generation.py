import logging
from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from frankstate.entity.runnable_builder import PromptMixin, RunnableBuilder
from utils.common import (
    load_and_clean_text_file,
    resolve_package_resource,
)


class MultimodalGeneration(PromptMixin, RunnableBuilder):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, model: BaseChatModel):
        super().__init__(model=model)

        self.logger.info("MultimodalGeneration initialized")

    def _build_prompt(self, **kwargs: Any) -> ChatPromptTemplate:
        docs_by_type = kwargs.get("context")
        question = kwargs.get("question")
        if not question or not isinstance(docs_by_type, dict):
            raise ValueError("Missing required keys 'question' and 'context' in kwargs")

        package = __package__ or __name__
        instructions = load_and_clean_text_file(
            resolve_package_resource(package, "prompt", "instructions.md")
        )

        format_template = load_and_clean_text_file(
            resolve_package_resource(package, "prompt", "format_template.md")
        )

        prompt_template = format_template.format(
            instructions=instructions,
            retrieved_context=docs_by_type["texts"],
            question=question,
        )

        prompt_content: list[str | dict[str, Any]] = [
            {"type": "text", "text": prompt_template}
        ]
        prompt_content.extend(docs_by_type["images"])

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content)])

    def _configure_runnable(self) -> Runnable[Any, Any]:
        rag_chain = (
            {
                "context": RunnableLambda(
                    lambda kwargs: cast(dict[str, Any], kwargs)["context"]
                ),
                "question": RunnableLambda(
                    lambda kwargs: cast(dict[str, Any], kwargs)["question"]
                ),
            }
            | RunnableLambda(lambda kwargs: self._build_prompt(**kwargs))
            | self.model
        )

        return rag_chain
