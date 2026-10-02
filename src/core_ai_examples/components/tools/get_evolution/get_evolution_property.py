from pydantic import BaseModel, Field


class GetEvolutionProperty:
    class Input(BaseModel):
        """Input for the GetEvolutionTool"""

        pokemon_name: str = Field(
            description="The name of the pokemon that want to know the evolutions."
        )

    name: str = "get_evolution"
    description: str = "This is a tool to obtain the evolutions of a pokemon."
    args_schema: type[BaseModel] = Input
    return_direct: bool = True
