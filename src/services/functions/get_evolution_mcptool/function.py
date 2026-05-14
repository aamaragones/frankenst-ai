import json

import azure.functions as func

from .orchestrator import Orchestrator
from .properties import tool_properties

bp_1 = func.Blueprint()

@bp_1.function_name(name="get_evolution_mcptool")
@bp_1.mcp_tool_trigger(
    arg_name="context",
    tool_name="get_evolution",
    description="Performs a simple addition of two integers.",
    tool_properties=json.dumps([prop.to_dict() for prop in tool_properties])
)
def main(context) -> str:
    content = json.loads(context)
    args = content.get("arguments", {})

    return str(Orchestrator.run(**args))
