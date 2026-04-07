from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from simpleneat.common import get_func_name

__all__ = ["export_onnx", "export_network_onnx"]


def export_onnx(
    state,
    genome,
    individual=None,
    index: int = 0,
    save_path: Optional[str] = None,
    metadata: Optional[Mapping[str, object]] = None,
    include_unused_nodes: bool = False,
    model_name: str = "simpleneat_network",
):
    """
    Export a feed-forward genome to ONNX for graph visualization tools such as Netron.

    This exporter focuses on producing a valid, readable ONNX graph. It is primarily
    intended for architecture inspection rather than guaranteeing numerical parity with
    every possible custom activation or aggregation function.
    """
    try:
        import onnx
        from onnx import TensorProto, checker, helper, numpy_helper
    except ImportError as exc:
        raise ImportError(
            "onnx is required for export_onnx(). Install it with `pip install onnx`."
        ) from exc

    if getattr(genome, "network_type", None) != "feedforward":
        raise NotImplementedError("ONNX export currently supports only feed-forward genomes.")
    input_transform = _get_transform_name(getattr(genome, "input_transform", None))
    if input_transform is not None and input_transform != "identity":
        raise NotImplementedError(
            "ONNX export does not yet support non-identity input transforms."
        )

    nodes, conns = _resolve_individual(state, individual, index)
    network = genome.network_dict(state, nodes, conns)

    input_idx = list(genome.get_input_idx())
    output_idx = list(genome.get_output_idx())
    selected_nodes = _select_nodes(network, input_idx, output_idx, include_unused_nodes)
    selected_conns = {
        (src_idx, dst_idx): conn_data
        for (src_idx, dst_idx), conn_data in network["conns"].items()
        if src_idx in selected_nodes and dst_idx in selected_nodes
    }
    topo_order = [
        node_idx
        for node_idx in network.get("topo_order", sorted(selected_nodes))
        if node_idx in selected_nodes
    ]

    onnx_nodes = []
    initializers = []
    tensor_names: dict[int, str] = {}

    graph_input_name = "inputs"
    graph_output_name = "outputs"

    onnx_inputs = [
        helper.make_tensor_value_info(graph_input_name, TensorProto.FLOAT, [len(input_idx)])
    ]
    onnx_outputs = [
        helper.make_tensor_value_info(graph_output_name, TensorProto.FLOAT, [len(output_idx)])
    ]

    for position, node_idx in enumerate(input_idx):
        index_name = f"input_index_{position}"
        output_name = f"input_node_{node_idx}"
        initializers.append(numpy_helper.from_array(np.asarray([position], dtype=np.int64), index_name))
        onnx_nodes.append(
            helper.make_node(
                "Gather",
                inputs=[graph_input_name, index_name],
                outputs=[output_name],
                axis=0,
                name=f"GatherInput_{node_idx}",
            )
        )
        if node_idx in selected_nodes:
            tensor_names[node_idx] = output_name

    incoming_edges: dict[int, list[tuple[int, dict]]] = {}
    for (src_idx, dst_idx), conn_data in sorted(selected_conns.items()):
        incoming_edges.setdefault(dst_idx, []).append((src_idx, conn_data))

    for node_idx in topo_order:
        if node_idx in input_idx:
            continue

        node_data = network["nodes"][node_idx]
        incoming = incoming_edges.get(node_idx, [])
        input_tensors = []

        for edge_idx, (src_idx, conn_data) in enumerate(incoming):
            if src_idx not in tensor_names:
                continue
            weight_name = f"weight_{src_idx}_{node_idx}_{edge_idx}"
            weighted_name = f"weighted_{src_idx}_{node_idx}_{edge_idx}"
            initializers.append(
                numpy_helper.from_array(
                    np.asarray([float(conn_data["weight"])], dtype=np.float32),
                    weight_name,
                )
            )
            onnx_nodes.append(
                helper.make_node(
                    "Mul",
                    inputs=[tensor_names[src_idx], weight_name],
                    outputs=[weighted_name],
                    name=f"MulWeight_{src_idx}_{node_idx}_{edge_idx}",
                )
            )
            input_tensors.append(weighted_name)

        aggregated_name = _aggregate_inputs(
            helper=helper,
            numpy_helper=numpy_helper,
            onnx_nodes=onnx_nodes,
            initializers=initializers,
            input_tensors=input_tensors,
            aggregation_name=str(node_data.get("agg", "sum")),
            node_idx=node_idx,
        )

        pre_activation_name = _apply_bias_and_response(
            helper=helper,
            numpy_helper=numpy_helper,
            onnx_nodes=onnx_nodes,
            initializers=initializers,
            input_name=aggregated_name,
            node_idx=node_idx,
            node_data=node_data,
        )

        if node_idx in output_idx:
            tensor_names[node_idx] = pre_activation_name
            continue

        tensor_names[node_idx] = _apply_activation(
            helper=helper,
            numpy_helper=numpy_helper,
            onnx_nodes=onnx_nodes,
            initializers=initializers,
            input_name=pre_activation_name,
            activation_name=str(node_data.get("act", "identity")),
            tensor_prefix=f"hidden_{node_idx}",
        )

    output_tensors = []
    for output_position, node_idx in enumerate(output_idx):
        if node_idx not in tensor_names:
            zero_name = f"missing_output_{output_position}"
            initializers.append(
                numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), zero_name)
            )
            output_tensors.append(zero_name)
            continue
        output_tensors.append(tensor_names[node_idx])

    if len(output_tensors) == 1:
        merged_output_name = output_tensors[0]
    else:
        merged_output_name = "concat_outputs"
        onnx_nodes.append(
            helper.make_node(
                "Concat",
                inputs=output_tensors,
                outputs=[merged_output_name],
                axis=0,
                name="ConcatOutputs",
            )
        )

    output_transform = _get_transform_name(getattr(genome, "output_transform", None))
    if output_transform is not None and output_transform != "identity":
        merged_output_name = _apply_activation(
            helper=helper,
            numpy_helper=numpy_helper,
            onnx_nodes=onnx_nodes,
            initializers=initializers,
            input_name=merged_output_name,
            activation_name=output_transform,
            tensor_prefix="output_transform",
        )

    if merged_output_name != graph_output_name:
        onnx_nodes.append(
            helper.make_node(
                "Identity",
                inputs=[merged_output_name],
                outputs=[graph_output_name],
                name="ModelOutput",
            )
        )

    graph = helper.make_graph(
        nodes=onnx_nodes,
        name=model_name,
        inputs=onnx_inputs,
        outputs=onnx_outputs,
        initializer=initializers,
    )
    model = helper.make_model(
        graph,
        producer_name="simpleneat",
        opset_imports=[helper.make_opsetid("", 11)],
    )

    if metadata:
        onnx.helper.set_model_props(
            model,
            {str(key): _stringify_metadata(value) for key, value in metadata.items()},
        )

    checker.check_model(model)

    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, output_path)
        print(f"ONNX export saved to {output_path}")

    return model


def export_network_onnx(*args, **kwargs):
    """Backward-compatible alias for ONNX export."""
    return export_onnx(*args, **kwargs)


def _resolve_individual(state, individual, index: int):
    if individual is not None:
        return individual

    if not hasattr(state, "pop_nodes") or not hasattr(state, "pop_conns"):
        raise ValueError(
            "State does not contain `pop_nodes`/`pop_conns`. Pass `individual=(nodes, conns)`."
        )

    return state.pop_nodes[index], state.pop_conns[index]


def _select_nodes(
    network: dict,
    input_idx: list[int],
    output_idx: list[int],
    include_unused_nodes: bool,
) -> set[int]:
    if include_unused_nodes:
        return set(network["nodes"])

    useful_nodes = set(network.get("useful_nodes", ()))
    if not useful_nodes:
        useful_nodes = set(network["nodes"])
    useful_nodes.update(input_idx)
    useful_nodes.update(output_idx)
    return useful_nodes


def _aggregate_inputs(
    helper,
    numpy_helper,
    onnx_nodes: list,
    initializers: list,
    input_tensors: list[str],
    aggregation_name: str,
    node_idx: int,
) -> str:
    if not input_tensors:
        zero_name = f"node_{node_idx}_zero"
        initializers.append(numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), zero_name))
        return zero_name

    if len(input_tensors) == 1:
        merged_name = input_tensors[0]
    else:
        merged_name = f"node_{node_idx}_concat"
        onnx_nodes.append(
            helper.make_node(
                "Concat",
                inputs=input_tensors,
                outputs=[merged_name],
                axis=0,
                name=f"ConcatInputs_{node_idx}",
            )
        )

    output_name = f"node_{node_idx}_agg"
    if aggregation_name == "sum":
        onnx_nodes.append(
            helper.make_node(
                "ReduceSum",
                inputs=[merged_name],
                outputs=[output_name],
                axes=[0],
                keepdims=1,
                name=f"ReduceSum_{node_idx}",
            )
        )
        return output_name
    if aggregation_name == "mean":
        onnx_nodes.append(
            helper.make_node(
                "ReduceMean",
                inputs=[merged_name],
                outputs=[output_name],
                axes=[0],
                keepdims=1,
                name=f"ReduceMean_{node_idx}",
            )
        )
        return output_name
    if aggregation_name == "product":
        onnx_nodes.append(
            helper.make_node(
                "ReduceProd",
                inputs=[merged_name],
                outputs=[output_name],
                axes=[0],
                keepdims=1,
                name=f"ReduceProd_{node_idx}",
            )
        )
        return output_name
    if aggregation_name == "max":
        onnx_nodes.append(
            helper.make_node(
                "ReduceMax",
                inputs=[merged_name],
                outputs=[output_name],
                axes=[0],
                keepdims=1,
                name=f"ReduceMax_{node_idx}",
            )
        )
        return output_name
    if aggregation_name == "min":
        onnx_nodes.append(
            helper.make_node(
                "ReduceMin",
                inputs=[merged_name],
                outputs=[output_name],
                axes=[0],
                keepdims=1,
                name=f"ReduceMin_{node_idx}",
            )
        )
        return output_name
    if aggregation_name == "maxabs":
        abs_name = f"node_{node_idx}_abs"
        argmax_name = f"node_{node_idx}_argmax"
        onnx_nodes.append(
            helper.make_node(
                "Abs",
                inputs=[merged_name],
                outputs=[abs_name],
                name=f"Abs_{node_idx}",
            )
        )
        onnx_nodes.append(
            helper.make_node(
                "ArgMax",
                inputs=[abs_name],
                outputs=[argmax_name],
                axis=0,
                keepdims=1,
                name=f"ArgMax_{node_idx}",
            )
        )
        onnx_nodes.append(
            helper.make_node(
                "Gather",
                inputs=[merged_name, argmax_name],
                outputs=[output_name],
                axis=0,
                name=f"GatherMaxAbs_{node_idx}",
            )
        )
        return output_name

    raise NotImplementedError(
        f"Unsupported aggregation `{aggregation_name}` for ONNX export."
    )


def _apply_bias_and_response(
    helper,
    numpy_helper,
    onnx_nodes: list,
    initializers: list,
    input_name: str,
    node_idx: int,
    node_data: dict,
) -> str:
    current_name = input_name
    response = float(node_data.get("response", 1.0))
    bias = float(node_data.get("bias", 0.0))

    if response != 1.0:
        response_name = f"node_{node_idx}_response"
        scaled_name = f"node_{node_idx}_scaled"
        initializers.append(
            numpy_helper.from_array(np.asarray([response], dtype=np.float32), response_name)
        )
        onnx_nodes.append(
            helper.make_node(
                "Mul",
                inputs=[current_name, response_name],
                outputs=[scaled_name],
                name=f"ApplyResponse_{node_idx}",
            )
        )
        current_name = scaled_name

    if bias != 0.0:
        bias_name = f"node_{node_idx}_bias"
        biased_name = f"node_{node_idx}_biased"
        initializers.append(numpy_helper.from_array(np.asarray([bias], dtype=np.float32), bias_name))
        onnx_nodes.append(
            helper.make_node(
                "Add",
                inputs=[current_name, bias_name],
                outputs=[biased_name],
                name=f"ApplyBias_{node_idx}",
            )
        )
        current_name = biased_name

    return current_name


def _apply_activation(
    helper,
    numpy_helper,
    onnx_nodes: list,
    initializers: list,
    input_name: str,
    activation_name: str,
    tensor_prefix: str,
) -> str:
    output_name = f"{tensor_prefix}_{activation_name}"

    if activation_name == "identity":
        onnx_nodes.append(
            helper.make_node(
                "Identity",
                inputs=[input_name],
                outputs=[output_name],
                name=f"Identity_{tensor_prefix}",
            )
        )
        return output_name
    if activation_name == "sigmoid":
        onnx_nodes.append(
            helper.make_node(
                "Sigmoid",
                inputs=[input_name],
                outputs=[output_name],
                name=f"Sigmoid_{tensor_prefix}",
            )
        )
        return output_name
    if activation_name == "tanh":
        onnx_nodes.append(
            helper.make_node(
                "Tanh",
                inputs=[input_name],
                outputs=[output_name],
                name=f"Tanh_{tensor_prefix}",
            )
        )
        return output_name
    if activation_name == "relu":
        onnx_nodes.append(
            helper.make_node(
                "Relu",
                inputs=[input_name],
                outputs=[output_name],
                name=f"Relu_{tensor_prefix}",
            )
        )
        return output_name
    if activation_name == "lelu":
        onnx_nodes.append(
            helper.make_node(
                "LeakyRelu",
                inputs=[input_name],
                outputs=[output_name],
                alpha=0.005,
                name=f"LeakyRelu_{tensor_prefix}",
            )
        )
        return output_name
    if activation_name == "sin":
        onnx_nodes.append(
            helper.make_node(
                "Sin",
                inputs=[input_name],
                outputs=[output_name],
                name=f"Sin_{tensor_prefix}",
            )
        )
        return output_name
    if activation_name == "abs":
        onnx_nodes.append(
            helper.make_node(
                "Abs",
                inputs=[input_name],
                outputs=[output_name],
                name=f"Abs_{tensor_prefix}",
            )
        )
        return output_name
    if activation_name == "exp":
        onnx_nodes.append(
            helper.make_node(
                "Exp",
                inputs=[input_name],
                outputs=[output_name],
                name=f"Exp_{tensor_prefix}",
            )
        )
        return output_name
    if activation_name == "log":
        onnx_nodes.append(
            helper.make_node(
                "Log",
                inputs=[input_name],
                outputs=[output_name],
                name=f"Log_{tensor_prefix}",
            )
        )
        return output_name
    if activation_name == "inv":
        onnx_nodes.append(
            helper.make_node(
                "Reciprocal",
                inputs=[input_name],
                outputs=[output_name],
                name=f"Reciprocal_{tensor_prefix}",
            )
        )
        return output_name
    if activation_name in {"scaled_sigmoid", "scaled_tanh"}:
        base_activation = "sigmoid" if activation_name == "scaled_sigmoid" else "tanh"
        base_output = _apply_activation(
            helper=helper,
            numpy_helper=numpy_helper,
            onnx_nodes=onnx_nodes,
            initializers=initializers,
            input_name=input_name,
            activation_name=base_activation,
            tensor_prefix=f"{tensor_prefix}_base",
        )
        scale_name = f"{tensor_prefix}_scale"
        initializers.append(
            numpy_helper.from_array(np.asarray([3.0], dtype=np.float32), scale_name)
        )
        onnx_nodes.append(
            helper.make_node(
                "Mul",
                inputs=[base_output, scale_name],
                outputs=[output_name],
                name=f"Scale_{tensor_prefix}",
            )
        )
        return output_name

    raise NotImplementedError(
        f"Unsupported activation `{activation_name}` for ONNX export."
    )


def _get_transform_name(transform) -> str | None:
    if transform is None:
        return None
    return get_func_name(transform)


def _stringify_metadata(value: object) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return repr(value)
