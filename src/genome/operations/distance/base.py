from simpleneat.common import StatefulBaseClass


class BaseDistance(StatefulBaseClass):
    def __call__(self, state, genome, nodes1, conns1, nodes2, conns2):
        raise NotImplementedError
