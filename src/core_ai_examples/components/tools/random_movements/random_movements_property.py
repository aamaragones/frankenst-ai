from pydantic import BaseModel, Field


class RandomMovementsProperty:
    class Input(BaseModel):
        """Input for the RandomMovementsTool"""

        pokemon_name: str = Field(
            description="The name of the pokemon that want to know random movements."
        )

    name: str = "random_movements"
    description: str = "This is a tool to obtain random movements of a pokemon."
    args_schema: type[BaseModel] = Input
    return_direct: bool = True
