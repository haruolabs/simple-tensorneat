from __future__ import annotations

from pathlib import Path
from typing import Optional

import networkx as nx


def draw(
    state,
    genome,
    individual=None,
    index: int = 0,
    save_path: Optional[str] = None,
    draw_weight_labels: bool = False,
    precision: int = 2,
    figsize: tuple[float, float] = (14.0, 10.0),
):
    """
    Draw a genome as an SVG/PNG-friendly matplotlib figure.

    Args:
        state: Current algorithm state.
        genome: Genome instance used to decode nodes/connections.
        individual: Optional ``(nodes, conns)`` tuple. If omitted, the function
            reads ``state.pop_nodes[index]`` and ``state.pop_conns[index]``.
        index: Population index used when ``individual`` is not provided.
        save_path: Optional output path such as ``network.svg`` or ``network.png``.
        draw_weight_labels: Whether to annotate edges with weights.
        precision: Decimal precision used for edge labels.
        figsize: Matplotlib figure size in inches.

    Returns:
        The created figure if ``save_path`` is not provided, otherwise ``None``.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for draw(). Install it with `pip install matplotlib`."
        ) from exc

    nodes, conns = _resolve_individual(state, individual, index)
    network = genome.network_dict(state, nodes, conns)
    graph = _build_graph(genome, network)

    if graph.number_of_nodes() == 0:
        raise ValueError("The selected genome does not contain any drawable nodes.")

    positions = _compute_positions(graph)
    fig, ax = plt.subplots(figsize=figsize)

    _draw_edges(ax, graph, positions, draw_weight_labels, precision)
    _draw_nodes(ax, graph, positions, patches)

    ax.set_aspect("equal")
    ax.axis("off")
    _set_axis_limits(ax, positions)
    _add_legend(ax, patches)
    plt.tight_layout()

    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Network visualization saved to {output}")
        return None

    return fig


def visualize_network(*args, **kwargs):
    """Backward-compatible alias for earlier experiments."""
    return draw(*args, **kwargs)


def _resolve_individual(state, individual, index: int):
    if individual is not None:
        return individual

    if not hasattr(state, "pop_nodes") or not hasattr(state, "pop_conns"):
        raise ValueError(
            "State does not contain `pop_nodes`/`pop_conns`. Pass `individual=(nodes, conns)`."
        )

    return state.pop_nodes[index], state.pop_conns[index]


def _build_graph(genome, network: dict) -> nx.DiGraph:
    graph = nx.DiGraph()
    input_idx = set(genome.get_input_idx())
    output_idx = set(genome.get_output_idx())
    useful_nodes = set(network.get("useful_nodes", ()))
    layer_lookup = {
        node_idx: layer_idx
        for layer_idx, layer in enumerate(network.get("topo_layers", []))
        for node_idx in layer
    }

    for idx, node_data in network["nodes"].items():
        role = _node_role(idx, input_idx, output_idx)
        graph.add_node(
            idx,
            label=_node_label(idx, role),
            role=role,
            color=_node_color(role, idx in useful_nodes),
            layer=layer_lookup.get(idx, 0),
            useful=idx in useful_nodes,
            **node_data,
        )

    for (src_idx, dst_idx), conn_data in network["conns"].items():
        graph.add_edge(
            src_idx,
            dst_idx,
            weight=float(conn_data["weight"]),
        )

    return graph


def _compute_positions(graph: nx.DiGraph) -> dict[int, tuple[float, float]]:
    layers: dict[int, list[tuple[int, dict]]] = {}
    for node_id, attrs in graph.nodes(data=True):
        layers.setdefault(int(attrs["layer"]), []).append((node_id, attrs))

    x_spacing = 3.6
    y_spacing = 2.1
    positions: dict[int, tuple[float, float]] = {}

    for layer_idx in sorted(layers):
        layer_nodes = sorted(
            layers[layer_idx],
            key=lambda item: (_role_order(item[1]["role"]), item[0]),
        )
        count = len(layer_nodes)
        y_offset = (count - 1) * y_spacing / 2

        for row_idx, (node_id, _) in enumerate(layer_nodes):
            x = layer_idx * x_spacing
            y = y_offset - row_idx * y_spacing
            positions[node_id] = (x, y)

    return positions


def _draw_nodes(ax, graph: nx.DiGraph, positions: dict[int, tuple[float, float]], patches):
    for node_id, attrs in graph.nodes(data=True):
        x, y = positions[node_id]
        circle = patches.Circle(
            (x, y),
            radius=0.24,
            facecolor=attrs["color"],
            edgecolor="black",
            linewidth=1.0 if attrs["useful"] else 0.75,
            zorder=2,
        )
        ax.add_patch(circle)
        ax.text(
            x,
            y,
            attrs["label"],
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            zorder=3,
        )


def _draw_edges(
    ax,
    graph: nx.DiGraph,
    positions: dict[int, tuple[float, float]],
    draw_weight_labels: bool,
    precision: int,
):
    for src_idx, dst_idx, attrs in graph.edges(data=True):
        src_pos = positions[src_idx]
        dst_pos = positions[dst_idx]
        weight = attrs["weight"]
        color = "#2e8b57" if weight > 0 else "#c0392b" if weight < 0 else "#7f8c8d"
        linewidth = max(1.0, min(4.0, 1.0 + abs(weight)))
        alpha = min(1.0, 0.35 + 0.2 * abs(weight))

        ax.annotate(
            "",
            xy=dst_pos,
            xytext=src_pos,
            arrowprops={
                "arrowstyle": "->",
                "color": color,
                "linewidth": linewidth,
                "alpha": alpha,
                "shrinkA": 18,
                "shrinkB": 18,
            },
            zorder=1,
        )

        if draw_weight_labels:
            mid_x = (src_pos[0] + dst_pos[0]) / 2
            mid_y = (src_pos[1] + dst_pos[1]) / 2
            ax.text(
                mid_x,
                mid_y,
                f"{weight:.{precision}f}",
                ha="center",
                va="center",
                fontsize=8,
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85},
                zorder=4,
            )


def _set_axis_limits(ax, positions: dict[int, tuple[float, float]]):
    xs = [coords[0] for coords in positions.values()]
    ys = [coords[1] for coords in positions.values()]
    x_pad = max(1.5, (max(xs) - min(xs)) * 0.18)
    y_pad = max(2.0, (max(ys) - min(ys)) * 0.3)
    ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)


def _add_legend(ax, patches):
    legend_elements = [
        patches.Patch(color="#9fd3ff", label="Input"),
        patches.Patch(color="#d9d9d9", label="Hidden"),
        patches.Patch(color="#9be7a1", label="Output"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")


def _node_label(idx: int, role: str) -> str:
    prefix = {"input": "I", "output": "O", "hidden": "H"}[role]
    return f"{prefix}{idx}"


def _node_color(role: str, useful: bool) -> str:
    if role == "input":
        return "#9fd3ff"
    if role == "output":
        return "#9be7a1"
    return "#d9d9d9" if useful else "#eeeeee"


def _node_role(idx: int, input_idx: set[int], output_idx: set[int]) -> str:
    if idx in input_idx:
        return "input"
    if idx in output_idx:
        return "output"
    return "hidden"


def _role_order(role: str) -> int:
    return {"input": 0, "hidden": 1, "output": 2}[role]
