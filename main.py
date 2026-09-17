"""Compile one of the reference layouts and print its Mermaid diagram.

python main.py --layout simple_oak [--with-metadata]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

LAYOUTS = ("simple_oak", "oak_human_loop", "local_vectorstore_rag", "ai_search_rag")


def _resolve(layout: str) -> tuple[type[Any], type[Any]]:
    from core_ai_examples.models.stategraph.ragstategraph import RAGState
    from core_ai_examples.models.stategraph.stategraph import SharedState

    if layout == "simple_oak":
        from config.graph_layout.simple_oak_config_graph import SimpleOakConfigGraph

        return SimpleOakConfigGraph, SharedState
    if layout == "oak_human_loop":
        from config.graph_layout.oak_human_loop_config_graph import (
            OakHumanLoopConfigGraph,
        )

        return OakHumanLoopConfigGraph, SharedState
    if layout == "local_vectorstore_rag":
        from config.graph_layout.local_vectorstore_adaptive_rag_config_graph import (
            LocalVectorStoreAdaptiveRAGConfigGraph,
        )

        return LocalVectorStoreAdaptiveRAGConfigGraph, RAGState
    from config.graph_layout.ai_search_adaptive_rag_config_graph import (
        AISearchAdaptiveRAGConfigGraph,
    )

    return AISearchAdaptiveRAGConfigGraph, RAGState


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", choices=LAYOUTS, default="simple_oak")
    parser.add_argument("--with-metadata", action="store_true")
    args = parser.parse_args(argv)

    from frankstate import WorkflowBuilder
    from utils.logger import configure_logging

    configure_logging()
    layout_cls, state_schema = _resolve(args.layout)
    builder = WorkflowBuilder(config=layout_cls, state_schema=state_schema)
    print(builder.to_mermaid(with_metadata=args.with_metadata))
    return 0


if __name__ == "__main__":
    sys.exit(main())
