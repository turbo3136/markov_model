import numpy as np
from markov_model.MarkovState import MarkovState


class MarkovStateSpace:
    """Create a MarkovStateSpace object defining the available state space

    Keyword arguments:
        state_id_list -- list of state_ids for each state
    """

    def __init__(self, state_id_list):
        self.state_id_list = state_id_list

        self.state_array = np.array([MarkovState(state_id=state_id) for state_id in self.state_id_list])
        self.size = len(self.state_array)

    def __repr__(self):
        return 'MarkovStateSpace(state_array={})'.format(self.state_array)
