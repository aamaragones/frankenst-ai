import pytest

from config.settings import get_settings
from core_ai_examples.components.tools.dominate_pokemon.dominate_pokemon_property import (
    DominatePokemonProperty,
)
from core_ai_examples.components.tools.dominate_pokemon.dominate_pokemon_tool import (
    DominatePokemonTool,
)
from core_ai_examples.components.tools.get_evolution.get_evolution_property import (
    GetEvolutionProperty,
)
from core_ai_examples.components.tools.get_evolution.get_evolution_tool import (
    GetEvolutionTool,
)
from core_ai_examples.components.tools.random_movements.random_movements_property import (
    RandomMovementsProperty,
)
from core_ai_examples.components.tools.random_movements.random_movements_tool import (
    RandomMovementsTool,
)
from utils.config_loader import load_node_registry

pytestmark = pytest.mark.unit

TOOLS = [GetEvolutionTool(), RandomMovementsTool(), DominatePokemonTool()]
PROPERTIES = [GetEvolutionProperty, RandomMovementsProperty, DominatePokemonProperty]


@pytest.mark.parametrize(("tool", "prop"), zip(TOOLS, PROPERTIES, strict=True))
def test_each_tool_takes_its_name_description_and_schema_from_its_property(
    tool: object, prop: type
) -> None:
    assert (tool.name, tool.description, tool.args_schema, tool.return_direct) == (  # type: ignore[attr-defined]
        prop.name,
        prop.description,
        prop.args_schema,
        prop.return_direct,
    )


def test_the_shipped_registry_names_exactly_the_local_tools() -> None:
    registry = load_node_registry(get_settings().config_nodes_file_path)

    assert set(registry["OAKTOOLS_NODE"]["metadata"]["tools"]) == {
        t.name for t in TOOLS
    }
