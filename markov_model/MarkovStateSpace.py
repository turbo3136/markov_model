class MarkovStateSpace:
    """Create a MarkovStateSpace object defining the available state space

    Keyword arguments:
        state_list -- list of MarkovState objects defining the state space
    """

    def __init__(self, state_list):
        self.state_list = state_list

    def __repr__(self):
        return 'MarkovStateSpace(state_id={})'.format(self.state_list)
