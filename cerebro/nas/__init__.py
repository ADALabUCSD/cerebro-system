from .tuners.randsearch import RandomSearch
from .tuners.hyperband import Hyperband
from .hphpmodel import HyperHyperModel
from .tuners.gridsearch import GridSearch

HyperHyperModel.__module__ = "cerebro.nas"
Hyperband.__module__ = "cerebro.nas.tuner"
GridSearch.__module__ = "cerebro.nas.tuner"
RandomSearch.__module__ = "cerebro.nas.tuner"