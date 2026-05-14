from pydantic import BaseModel


class LangChainToolProperty:
    def __init__(
        self,
        input: type[BaseModel],
        description: str,
        return_direct: bool = True,
    ):
        self.args_schema: type[BaseModel] = input
        self.description: str = description
        self.return_direct: bool = return_direct
