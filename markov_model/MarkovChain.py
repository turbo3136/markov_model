import numpy as np
import copy


class MarkovChain:
    """Create a MarkovChain object based on an initial state, state space, and transition matrix

    Keyword arguments:
        initial_state -- MarkovStateVector object representing the initial state of the chain
        state_space -- MarkovStateSpace object representing the state space, i.e. all possible MarkovState(s)
        transition_matrix -- MarkovTransitionMatrix object representing the transition matrix and its functions
        total_steps -- int representing the number of states we'll calculate past the initial time_step

    Properties:
        history -- array of states that the chain has occupied
        current_state -- MarkovStateVector object representing the current state of the chain after total_steps

    Methods:
        next_state(starting_state, log_history, make_deepcopy) --
            return the state after starting_state
            if requested, log the history in self.history, default is False
            if requested, make a deep copy of the starting_state, default is True

        state_after_n_steps(starting_state, n, log_history) --
            return the state n steps after starting_state
            if requested, log the history in self.history, default is False

        vectorized_state_after_n_steps(starting_state, n, log_history) --
            same as state_after_n_steps, but allows for one of the arguments to be an array input
    """

    def __init__(self, initial_state, state_space, transition_matrix, total_steps):
        # do a few value checks on the sizes of the states and matrix
        if state_space.size != len(initial_state.state_distribution):
            raise ValueError(
                'MarkovChain.state_space must be the same size as MarkovChain.initial_state.state_distribution'
            )
        if state_space.size != transition_matrix.transition_function_matrix.shape[0]:
            raise ValueError(
                '''MarkovChain.transition_matrix must be a square matrix 
                with each dimension the size of MarkovChain.state_space'''
            )

        self.initial_state = initial_state  # reminder: this is a MarkovStateVector object
        self.state_space = state_space  # reminder: this is a MarkovStateSpace object
        self.transition_matrix = transition_matrix  # reminder: this is a MarkovTransitionMatrix object
        self.total_steps = total_steps
        self.history = np.array([self.initial_state])  # initialize the history with the initial state

        # then we calculate the current state and log the history along the way
        self.current_state = self.state_after_n_steps(self.initial_state, self.total_steps, log_history=True)

    def __repr__(self):
        return 'MarkovChain(initial_state={}, state_space={})'.format(self.initial_state, self.state_space)

    def next_state(self, starting_state, log_history=False, make_deepcopy=True):
        """return a MarkovStateVector object after applying the transition matrix"""
        if make_deepcopy:  # make a copy of the starting MarkovStateVector so we can update it
            next_state = copy.deepcopy(starting_state)
        else:  # otherwise, just add a new reference
            next_state = starting_state

        next_state.time_step += 1  # add 1 to the starting time_step

        # grab the starting state distribution and take the dot product of the transition matrix at the time_step
        next_state.state_distribution = starting_state.state_distribution.dot(
            self.transition_matrix.matrix_at_time_step(starting_state.time_step)
        )

        # check to see if we want to log the history
        if log_history:
            self.history = np.append(self.history, next_state)  # add the next state to the history log

        return next_state

    def state_after_n_steps(self, starting_state, n, log_history=False):
        """return a MarkovStateVector object after applying n transitions"""
        current_state = copy.deepcopy(starting_state)

        for step in np.arange(n):
            current_state = self.next_state(current_state, log_history, make_deepcopy=False)

        return current_state

    def vectorized_state_after_n_steps(self, starting_state, n, log_history=False):
        """return a MarkovStateVector object after applying n transitions

        vectorize to allow for array inputs for either starting_state or n, not both
        """
        vectorized_func = np.vectorize(self.state_after_n_steps)
        return vectorized_func(starting_state, n, log_history)
