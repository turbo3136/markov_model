class MarkovStateVector:
    """Create a MarkovStateVector object defining the available state space

    Keyword arguments:
        state_list -- list of MarkovState objects defining the state space
        time_step -- time step of the current vector
    """

    def __init__(self, state_list, time_step):
        self.state_list = state_list
        self.time_step = time_step

    def __repr__(self):
        return 'MarkovStateVector(state_id={}, time_step={})'.format(self.state_list, self.time_step)
