import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool

from core_examples.utils.common import (
    load_and_clean_text_file,
    resolve_package_resource,
)
from frankstate.entity.runnable_builder import PromptMixin, RunnableBuilder

from .history_template import history_template


class OakLangAgent(PromptMixin, RunnableBuilder):
    logger: logging.Logger = logging.getLogger(__name__)

    def __init__(self, model: BaseChatModel, tools: list[BaseTool]):
        self.tools = tools
        super().__init__(model=model)

        self.logger.info("OakLangAgent initialized")

    def _build_prompt(self) -> ChatPromptTemplate:
        # Prepare the few_shot_prompt
        # few_shot_map = ChatPromptTemplate.from_messages(
        #     [
        #         ("human", "{input}"),
        #         ("ai", "{output}"),
        #     ]
        # )
        # few_shot_prompt = FewShotChatMessagePromptTemplate(
        #     example_prompt=few_shot_map,
        #     examples=few_shot_examples,
        # )

        # Prepare the prompt
        package = __package__ or __name__
        context = load_and_clean_text_file(resolve_package_resource(package, 'prompt', 'context.md'))
        instructions = load_and_clean_text_file(resolve_package_resource(package, 'prompt', 'instructions.md'))
        input = load_and_clean_text_file(resolve_package_resource(package, 'prompt', 'input.md'))
        output_format = load_and_clean_text_file(resolve_package_resource(package, 'prompt', 'output_format.md'))
        restrictions = load_and_clean_text_file(resolve_package_resource(package, 'prompt', 'restrictions.md'))

        format_template = load_and_clean_text_file(resolve_package_resource(package, 'prompt', 'format_template.md'))

        system_prompt = format_template.format(
            context=context,
            instructions=instructions,
            input=input,
            output_format=output_format,
            restrictions=restrictions
        )

        self.logger.info(system_prompt)

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            # few_shot_prompt,
            history_template[0],
            MessagesPlaceholder(variable_name="messages"),
        ])

    def _configure_runnable(self) -> Runnable:
        prompt_template = self._build_prompt()
        model_with_tools = self.model.bind_tools(self.tools or [])

        chain = prompt_template | model_with_tools

        return chain