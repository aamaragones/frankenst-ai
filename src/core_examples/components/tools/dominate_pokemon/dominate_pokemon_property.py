from dataclasses import dataclass

from pydantic import BaseModel, Field


class DominatePokemonProperty:
    @dataclass
    class Input(BaseModel):
        """
        Input for the DominatePokemonTool
        """
        place: str = Field(description="Place could be a city, country or continent")

    description: str = "This is a tool that move trainers to capture and/or dominate all the pokemon of a certain place. Return to a user an url to see the effect of that action."
    args_schema: type[BaseModel] = Input
    return_direct: bool = True