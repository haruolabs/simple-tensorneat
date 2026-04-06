from typing import Callable

from simpleneat.common import State, StatefulBaseClass


class BaseProblem(StatefulBaseClass):
    jitable = None
    supports_backprop = False

    def evaluate(self, state: State, randkey, act_func: Callable, params):
        raise NotImplementedError

    @property
    def input_shape(self):
        raise NotImplementedError

    @property
    def output_shape(self):
        raise NotImplementedError

    def show(self, state: State, randkey, act_func: Callable, params, *args, **kwargs):
        raise NotImplementedError

    def show_details(self, state: State, randkey, act_func: Callable, pop_params, *args, **kwargs):
        return None
