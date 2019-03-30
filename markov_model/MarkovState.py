class MarkovState:
    """Create a MarkovState object with identifying attributes

    Keyword arguments:
        state_id -- unique identifier
        state_name -- optional, display name. Set to state_id if None
        state_description -- optional, readable description of the state
    """

    def __init__(self, state_id, state_name=None, state_description=None):
        if not state_name:  # if we weren't provided a state_name, set it to state_id
            state_name = state_id

        self.state_id = state_id
        self.state_name = state_name
        self.state_description = state_description

    def __repr__(self):
        return 'MarkovState(state_id={}, state_name={})'.format(self.state_id, self.state_name)
