import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from core_examples.utils.common import (
    load_and_clean_text_file,
    resolve_package_resource,
)
from frankstate.entity.runnable_builder import RunnableBuilder


class RewriteQuestion(RunnableBuilder):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, model: BaseChatModel):
        super().__init__(model=model)

        self.logger.info("RewriteQuestion initialized")

    def _build_prompt(self, kwargs: dict[str, Any]) -> ChatPromptTemplate:
        question = kwargs["question"]

        # Prepare the human_prompt
        package = __package__ or __name__
        format_template = load_and_clean_text_file(resolve_package_resource(package, 'prompt', 'format_template.md'))

        prompt_template = format_template.format(
            question=question
        )

        prompt_content: list[str | dict[str, Any]] = [{"type": "text", "text": prompt_template}]

        return ChatPromptTemplate.from_messages([
            HumanMessage(content=prompt_content)
        ])

    def _configure_runnable(self) -> Runnable:
        rewrite_chain = {
            "question": RunnablePassthrough(),
        } | RunnableLambda(self._build_prompt) | self.model

        return rewrite_chain