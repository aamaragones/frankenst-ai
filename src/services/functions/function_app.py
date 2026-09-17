import azure.functions as func

try:
    from get_evolution_mcptool.function import bp_1
    from indexer_pokeseriex_eventgrid.function import bp_2
    from retriever_pokeseriex_mcptool.function import bp_3
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "This module is part of an Azure Functions packaging artifact "
        "and is not importable directly from the source tree. Use "
        "the container flow defined in src/services/functions/Dockerfile."
    ) from exc


app = func.FunctionApp()

# Register all the Blueprint apps
bps = [
    bp_1,
    bp_2,
    bp_3,
]
for bp in bps:
    app.register_functions(bp)
