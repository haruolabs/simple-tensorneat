from simpleneat.common import State, StatefulBaseClass


class BaseAlgorithm(StatefulBaseClass):
    def ask(self, state: State):
        raise NotImplementedError

    def tell(self, state: State, fitness):
        raise NotImplementedError

    def transform(self, state, individual):
        raise NotImplementedError

    def forward(self, state, transformed, inputs):
        raise NotImplementedError

    def adapt(self, state: State, problem):
        return state

    def show_details(self, state: State, fitness):
        return None

    @property
    def num_inputs(self):
        raise NotImplementedError

    @property
    def num_outputs(self):
        raise NotImplementedError
