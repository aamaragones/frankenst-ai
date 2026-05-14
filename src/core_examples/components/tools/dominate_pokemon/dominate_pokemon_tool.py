import logging

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, SkipValidation

from .dominate_pokemon import DominatePokemon
from .dominate_pokemon_property import DominatePokemonProperty


class DominatePokemonTool(BaseTool):
    # BaseTool atributes
    name: str | None = None
    description: str | None = None
    args_schema: type[BaseModel] | None = None
    return_direct: bool | None = None

    # New BaseTool attributes
    config: SkipValidation[DominatePokemonProperty] = DominatePokemonProperty()
    logger: SkipValidation[logging.Logger] = logging.getLogger(__name__)
     
    def __init__(self, **data):
        super().__init__(**data)
        self.name = self.__class__.__name__
        self.description = self.config.description
        self.args_schema = self.config.args_schema
        self.return_direct = self.config.return_direct
        
        
    def _run(self, place: str, run_manager: CallbackManagerForToolRun | None = None) -> list[str]:
        """
        Run the tool logic
        """
        self.logger.info(f"Args: {place}")
        
        return DominatePokemon.run(place=place)