import json

import azure.functions as func

from .orchestrator import Orchestrator
from .properties import tool_properties

bp_3 = func.Blueprint()

@bp_3.function_name(name="retriever_pokeseriex_mcptool")
@bp_3.mcp_tool_trigger(
    arg_name="context",
    tool_name="retriever_pokeseriex",
    description="Esta herramienta transforma una consulta en lenguaje natural en un vector y recupera contexto del contenido y los esquemas de una base de datos, Teradata.",
    tool_properties=json.dumps([prop.to_dict() for prop in tool_properties])
)
def main(context) -> str:
    content = json.loads(context)
    args = content.get("arguments", {})

    return Orchestrator.run(**args)
