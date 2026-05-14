from dataclasses import dataclass

from pydantic import BaseModel, Field


class GetEvolutionProperty:
    @dataclass
    class Input(BaseModel):
        """
        Input for the GetEvolutionTool
        """
        pokemon_name: str = Field(description="The name of the pokemon that want to know the evolutions.")

    description: str = "This is a tool to obtain the evolutions of a pokemon."
    args_schema: type[BaseModel] = Input
    return_direct: bool = True