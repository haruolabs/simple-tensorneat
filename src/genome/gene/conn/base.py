from ..base import BaseGene


class BaseConn(BaseGene):
    fixed_attrs = ["input_index", "output_index"]

    def new_zero_attrs(self, state):
        raise NotImplementedError

    def forward(self, state, attrs, inputs):
        raise NotImplementedError

    def repr(self, state, conn, precision=2, idx_width=3, func_width=8):
        in_idx, out_idx = conn[:2]
        return (
            f"{self.__class__.__name__}(in={int(in_idx):<{idx_width}}, "
            f"out={int(out_idx):<{idx_width}})"
        )

    def to_dict(self, state, conn):
        return {
            "in": int(conn[0]),
            "out": int(conn[1]),
        }

