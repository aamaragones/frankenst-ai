from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, cast

import yaml


def _parse_yaml(data: dict[str, Any]) -> dict[str, Any]:
	def resolve_value(value: Any, context: dict[str, Any]) -> Any:
		if isinstance(value, str) and "$(" in value:
			while "$(" in value:
				start_index = value.find("$(")
				end_index = value.find(")", start_index)
				if end_index == -1:
					raise ValueError(f"Unmatched $() in value: {value}")

				ref_key = value[start_index + 2:end_index]
				ref_value = get_nested_value(context, ref_key.split("."))
				if ref_value is None:
					raise KeyError(f"Reference '{ref_key}' not found in the context.")

				value = value[:start_index] + str(ref_value) + value[end_index + 1:]
			return value
		if isinstance(value, dict):
			return {key: resolve_value(item, context) for key, item in value.items()}
		if isinstance(value, list):
			return [resolve_value(item, context) for item in value]
		return value

	def get_nested_value(data: dict[str, Any], keys: list[str]) -> Any:
		for key in keys:
			if not isinstance(data, dict) or key not in data:
				return None
			data = data[key]
		return data

	return cast(dict[str, Any], resolve_value(data, data))


def _read_text_resource(resource_path: str | Path | Traversable, encoding: str = "utf-8") -> str:
	if isinstance(resource_path, Traversable):
		return resource_path.read_text(encoding=encoding)

	return Path(resource_path).expanduser().read_text(encoding=encoding)


def read_yaml(path_to_yaml: str | Path | Traversable) -> dict[str, Any]:
	try:
		yaml_content = yaml.safe_load(_read_text_resource(path_to_yaml))
		if yaml_content is None:
			raise ValueError(f"The configuration file '{path_to_yaml}' is empty.")
		if not isinstance(yaml_content, dict):
			raise ValueError(
				f"The configuration file '{path_to_yaml}' must contain a YAML mapping at the root."
			)

		parsed_yaml_context = _parse_yaml(yaml_content)
		if not parsed_yaml_context:
			raise ValueError(f"The configuration file '{path_to_yaml}' must not be an empty mapping.")

		return parsed_yaml_context
	except FileNotFoundError:
		raise FileNotFoundError(f"The configuration file '{path_to_yaml}' does not exist.") from None
	except yaml.YAMLError as exc:
		raise yaml.YAMLError(f"Error parsing YAML file {exc}") from exc
	except (KeyError, ValueError):
		raise
	except OSError as exc:
		raise RuntimeError(f"Unable to read configuration file '{path_to_yaml}'") from exc


def load_node_registry(path_to_yaml: str | Path | Traversable) -> dict[str, dict[str, Any]]:
	try:
		data = yaml.safe_load(_read_text_resource(path_to_yaml))
		if data is None:
			raise ValueError(f"The configuration file '{path_to_yaml}' is empty.")

		if not isinstance(data, dict):
			raise ValueError(f"The node registry '{path_to_yaml}' must contain a YAML mapping at the root.")

		nodes = data.get("nodes")
		if not isinstance(nodes, list) or not nodes:
			raise ValueError(f"The node registry '{path_to_yaml}' must define a non-empty top-level 'nodes' list.")

		validated_nodes: list[dict[str, Any]] = []
		required_fields = {"id", "name", "type", "description"}
		for entry in nodes:
			if not isinstance(entry, dict):
				raise ValueError(f"The node registry '{path_to_yaml}' must contain only mapping entries inside 'nodes'.")

			missing_fields = sorted(field for field in required_fields if not entry.get(field))
			if missing_fields:
				raise ValueError(
					f"The node registry '{path_to_yaml}' is missing required node fields: {missing_fields}"
				)

			destinations = entry.get("destinations")
			if destinations is not None:
				if not isinstance(destinations, dict) or not all(
					isinstance(key, str) and isinstance(value, str)
					for key, value in destinations.items()
				):
					raise ValueError(
						f"The node registry '{path_to_yaml}' must define 'destinations' as a string-to-string mapping."
					)

			validated_nodes.append(entry)

		return {
			node["id"]: {key: value for key, value in node.items() if key != "id"}
			for node in validated_nodes
		}
	except FileNotFoundError:
		raise FileNotFoundError(f"The configuration file '{path_to_yaml}' does not exist.") from None
	except yaml.YAMLError as exc:
		raise yaml.YAMLError(f"Error parsing YAML file {exc}") from exc