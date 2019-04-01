import numpy as np


class MarkovStateVector:
    """Create a MarkovStateVector object defining a probability distribution within the state space.

    Keyword arguments:
        state_space -- MarkovStateSpace object for this system, i.e. a list of all possible MarkovState(s)
        state_distribution -- numpy array representing the probability distribution within the state space
        time_step -- time step for the system this vector represents
    """

    def __init__(self, state_space, state_distribution, time_step):
        if type(state_distribution) != np.ndarray:
            raise ValueError('MarkovStateVector.state_distribution expects a numpy ndarray object')

        self.state_space = state_space
        self.state_distribution = state_distribution
        self.time_step = time_step

        if self.state_space.size != len(self.state_distribution):
            raise ValueError(
                'MarkovStateVector.state_distribution must be the same size as state_space array'
            )

    def __repr__(self):
        return 'MarkovStateVector(state_space={}, state_distribution={}, time_step={})'.format(
            self.state_space, self.state_distribution, self.time_step
        )
