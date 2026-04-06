from ..base import BaseGene


class BaseNode(BaseGene):
    fixed_attrs = ["index"]

    def forward(self, state, attrs, inputs, is_output_node=False, valid_mask=None):
        raise NotImplementedError

    def repr(self, state, node, precision=2, idx_width=3, func_width=8):
        idx = int(node[0])
        return f"{self.__class__.__name__}(idx={idx:<{idx_width}})"

    def to_dict(self, state, node):
        return {"idx": int(node[0])}

