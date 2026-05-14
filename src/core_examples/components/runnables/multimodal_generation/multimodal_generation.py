import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda

from core_examples.utils.common import (
    load_and_clean_text_file,
    resolve_package_resource,
)
from frankstate.entity.runnable_builder import RunnableBuilder


class MultimodalGeneration(RunnableBuilder):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, model: BaseChatModel):
        super().__init__(model=model)

        self.logger.info("MultimodalGeneration initialized")

    def _build_prompt(self, kwargs: dict[str, Any]) -> ChatPromptTemplate:
        docs_by_type = kwargs["context"]
        question = kwargs["question"]

        # Prepare the human_prompt
        package = __package__ or __name__
        instructions = load_and_clean_text_file(resolve_package_resource(package, 'prompt', 'instructions.md'))

        format_template = load_and_clean_text_file(resolve_package_resource(package, 'prompt', 'format_template.md'))

        prompt_template = format_template.format(
            instructions=instructions,
            retrieved_context=docs_by_type["texts"],
            question=question
        )
        
        prompt_content: list[str | dict[str, Any]] = [{"type": "text", "text": prompt_template}]
        prompt_content.extend(docs_by_type["images"])

        return ChatPromptTemplate.from_messages([
            HumanMessage(content=prompt_content)
        ])

    def _configure_runnable(self) -> Runnable:
        rag_chain = {
            "context": RunnableLambda(lambda kwargs: kwargs["context"]),
            "question": RunnableLambda(lambda kwargs: kwargs["question"]),
        } | RunnableLambda(self._build_prompt) | self.model

        return rag_chain
