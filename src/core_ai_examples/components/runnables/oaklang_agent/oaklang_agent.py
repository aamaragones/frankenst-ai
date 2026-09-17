import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from frankstate.entity.runnable_builder import PromptMixin, RunnableBuilder
from utils.common import load_and_clean_text_file, resolve_package_resource

from .history_template import history_template

_FRAGMENTS = ("context", "instructions", "input", "output_format", "restrictions")


class OakLangAgent(PromptMixin, RunnableBuilder):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, model: BaseChatModel, tools: list[BaseTool]):
        self.tools = tools
        super().__init__(model=model)
        self.logger.info("OakLangAgent initialized")

    @staticmethod
    def _fragment(name: str) -> str:
        package = __package__ or __name__
        return load_and_clean_text_file(
            resolve_package_resource(package, "prompt", f"{name}.md")
        )

    def _build_prompt(self, **kwargs: Any) -> ChatPromptTemplate:
        system_prompt = self._fragment("format_template").format(
            **{name: self._fragment(name) for name in _FRAGMENTS}
        )
        self.logger.debug(system_prompt)
        return ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                history_template[0],
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

    def _configure_runnable(self) -> Runnable[Any, Any]:
        return self._build_prompt() | self.model.bind_tools(self.tools or [])
