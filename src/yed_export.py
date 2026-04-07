from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional

from simpleneat.common import get_func_name

__all__ = ["export_yed_json", "export_network_yed_json"]


def export_yed_json(
    state,
    genome,
    individual=None,
    index: int = 0,
    save_path: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
    include_unused_nodes: bool = True,
    graph_id: str = "simpleneat_network",
):
    """
    Export a genome as a simple node/edge JSON document that yFiles-based tools can ingest.

    The payload intentionally keeps the graph structure flat and explicit:
    `nodes[*].id`, `edges[*].start`, and `edges[*].end` define the topology, while
    labels and NEAT-specific details live in each item's `properties`.
    """
    nodes, conns = _resolve_individual(state, individual, index)
    network = genome.network_dict(state, nodes, conns)

    input_idx = set(genome.get_input_idx())
    output_idx = set(genome.get_output_idx())
    useful_nodes = set(network.get("useful_nodes", ()))
    output_transform = _get_transform_name(getattr(genome, "output_transform", None))

    if include_unused_nodes:
        selected_nodes = set(network["nodes"])
    else:
        selected_nodes = useful_nodes | input_idx | output_idx

    selected_conns = {
        (src_idx, dst_idx): conn_data
        for (src_idx, dst_idx), conn_data in network["conns"].items()
        if src_idx in selected_nodes and dst_idx in selected_nodes
    }

    layer_lookup = {
        node_idx: layer_idx
        for layer_idx, layer in enumerate(network.get("topo_layers", []))
        for node_idx in layer
    }
    topo_order = [
        node_idx
        for node_idx in network.get("topo_order", sorted(selected_nodes))
        if node_idx in selected_nodes
    ]
    remaining_nodes = sorted(selected_nodes.difference(topo_order))
    ordered_nodes = topo_order + remaining_nodes

    payload = {
        "id": graph_id,
        "directed": True,
        "multigraph": False,
        "metadata": {
            "input_count": len(input_idx),
            "output_count": len(output_idx),
            "include_unused_nodes": include_unused_nodes,
            "output_transform": output_transform,
            **_jsonify_mapping(metadata or {}),
        },
        "nodes": [
            _build_node_record(
                node_idx=node_idx,
                node_data=network["nodes"][node_idx],
                input_idx=input_idx,
                output_idx=output_idx,
                useful_nodes=useful_nodes,
                layer_lookup=layer_lookup,
                output_transform=output_transform,
            )
            for node_idx in ordered_nodes
        ],
        "edges": [
            _build_edge_record(src_idx, dst_idx, conn_data)
            for (src_idx, dst_idx), conn_data in sorted(selected_conns.items())
        ],
    }

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=False)
            fp.write("\n")
        print(f"yEd JSON export saved to {output_path}")

    return payload


def export_network_yed_json(*args, **kwargs):
    """Backward-compatible alias for yEd JSON export."""
    return export_yed_json(*args, **kwargs)


def _resolve_individual(state, individual, index: int):
    if individual is not None:
        return individual

    if not hasattr(state, "pop_nodes") or not hasattr(state, "pop_conns"):
        raise ValueError(
            "State does not contain `pop_nodes`/`pop_conns`. Pass `individual=(nodes, conns)`."
        )

    return state.pop_nodes[index], state.pop_conns[index]


def _build_node_record(
    node_idx: int,
    node_data: Mapping[str, object],
    input_idx: set[int],
    output_idx: set[int],
    useful_nodes: set[int],
    layer_lookup: Mapping[int, int],
    output_transform: Optional[str],
) -> dict[str, object]:
    role = _node_role(node_idx, input_idx, output_idx)
    label = _node_label(node_idx, role, node_data, output_transform)
    properties = {
        "idx": int(node_idx),
        "label": label,
        "role": role,
        "layer": int(layer_lookup.get(node_idx, 0)),
        "useful": node_idx in useful_nodes,
        "aggregation": node_data.get("agg"),
        "activation": node_data.get("act"),
        "bias": _jsonify_value(node_data.get("bias")),
        "response": _jsonify_value(node_data.get("response")),
    }
    if role == "output":
        properties["output_transform"] = output_transform

    return {
        "id": int(node_idx),
        "label": label,
        "properties": properties,
    }


def _build_edge_record(
    src_idx: int,
    dst_idx: int,
    conn_data: Mapping[str, object],
) -> dict[str, object]:
    weight = float(conn_data["weight"])
    return {
        "id": f"{src_idx}->{dst_idx}",
        "start": int(src_idx),
        "end": int(dst_idx),
        "label": f"{weight:.3f}",
        "properties": {
            "label": f"{weight:.3f}",
            "weight": weight,
        },
    }


def _node_label(
    node_idx: int,
    role: str,
    node_data: Mapping[str, object],
    output_transform: Optional[str],
) -> str:
    if role == "input":
        return f"Input {node_idx}"
    if role == "output":
        if output_transform and output_transform != "identity":
            return f"Output {node_idx}\n{output_transform}"
        return f"Output {node_idx}"
    activation = str(node_data.get("act", "identity"))
    return f"Hidden {node_idx}\n{activation}"


def _node_role(node_idx: int, input_idx: set[int], output_idx: set[int]) -> str:
    if node_idx in input_idx:
        return "input"
    if node_idx in output_idx:
        return "output"
    return "hidden"


def _get_transform_name(transform) -> Optional[str]:
    if transform is None:
        return None
    return get_func_name(transform)


def _jsonify_mapping(mapping: Mapping[str, object]) -> dict[str, object]:
    return {str(key): _jsonify_value(value) for key, value in mapping.items()}


def _jsonify_value(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return _jsonify_mapping(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_jsonify_value(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    return str(value)
