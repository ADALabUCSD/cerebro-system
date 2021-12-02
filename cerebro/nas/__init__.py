from autokeras.tuners.greedy import Greedy

from .tuners.greedy import GreedySearch
from .tuners.randsearch import RandomSearch
# from .tuners.hyperband import Hyperband
from .hphpmodel import HyperHyperModel
from .tuners.gridsearch import GridSearch
from .tuners.greedy import GreedySearch

HyperHyperModel.__module__ = "cerebro.nas"
# Hyperband.__module__ = "cerebro.nas.tuner"
GridSearch.__module__ = "cerebro.nas.tuner"
RandomSearch.__module__ = "cerebro.nas.tuner"
GreedySearch.__module__ = "cerebro.nas.greedy"