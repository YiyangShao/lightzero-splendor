# Stub module for missing C++ extension
# This is a placeholder to prevent import errors when C++ extensions are not compiled

class StubClass:
    """Stub class to prevent import errors"""
    def __init__(self, *args, **kwargs):
        raise ImportError("C++ extension 'ctree_muzero' is not available. Please compile the C++ extensions or use Python MCTS implementation.")

# Create stub classes for all expected classes
Roots = StubClass
MinMaxStatsList = StubClass
ResultsWrapper = StubClass

def batch_traverse(*args, **kwargs):
    raise ImportError("C++ extension 'ctree_muzero' is not available. Please compile the C++ extensions or use Python MCTS implementation.")

def batch_backpropagate(*args, **kwargs):
    raise ImportError("C++ extension 'ctree_muzero' is not available. Please compile the C++ extensions or use Python MCTS implementation.")
