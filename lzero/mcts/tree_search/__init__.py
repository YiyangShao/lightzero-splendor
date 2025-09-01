# Conditional imports to avoid C++ extension dependencies
try:
    from .mcts_ctree import MuZeroMCTSCtree, EfficientZeroMCTSCtree, GumbelMuZeroMCTSCtree, UniZeroMCTSCtree, MuZeroRNNFullObsMCTSCtree
except ImportError:
    # C++ extensions not available, define placeholder classes
    MuZeroMCTSCtree = None
    EfficientZeroMCTSCtree = None
    GumbelMuZeroMCTSCtree = None
    UniZeroMCTSCtree = None
    MuZeroRNNFullObsMCTSCtree = None

try:
    from .mcts_ctree_sampled import SampledEfficientZeroMCTSCtree, SampledMuZeroMCTSCtree, SampledUniZeroMCTSCtree
except ImportError:
    SampledEfficientZeroMCTSCtree = None
    SampledMuZeroMCTSCtree = None
    SampledUniZeroMCTSCtree = None

try:
    from .mcts_ctree_stochastic import StochasticMuZeroMCTSCtree
except ImportError:
    StochasticMuZeroMCTSCtree = None

try:
    from .mcts_ptree import MuZeroMCTSPtree, EfficientZeroMCTSPtree
except ImportError:
    MuZeroMCTSPtree = None
    EfficientZeroMCTSPtree = None

try:
    from .mcts_ptree_sampled import SampledEfficientZeroMCTSPtree
except ImportError:
    SampledEfficientZeroMCTSPtree = None

try:
    from .mcts_ptree_stochastic import StochasticMuZeroMCTSPtree
except ImportError:
    StochasticMuZeroMCTSPtree = None
