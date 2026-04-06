from simpleneat.common import StatefulBaseClass


class BaseCrossover(StatefulBaseClass):
    def __call__(self, state, genome, randkey, nodes1, conns1, nodes2, conns2):
        raise NotImplementedError
