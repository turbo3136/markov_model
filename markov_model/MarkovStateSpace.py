import numpy as np


class MarkovStateSpace:
    """Create a MarkovStateSpace object defining the available state space

    Keyword arguments:
        state_array -- numpy array of MarkovState objects defining the state space
    """

    def __init__(self, state_array):
        if type(state_array) != np.ndarray:
            raise ValueError('MarkovStateSpace.state_array expects a numpy ndarray object')
        self.state_array = state_array
        self.size = len(self.state_array)

    def __repr__(self):
        return 'MarkovStateSpace(state_array={})'.format(self.state_array)
