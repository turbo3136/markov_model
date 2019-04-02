import numpy as np
import copy


class MarkovChain:
    """Create a MarkovChain object based on an initial state, state space, and transition matrix

    Keyword arguments:
        initial_state -- MarkovStateVector object representing the initial state of the chain
        state_space -- MarkovStateSpace object representing the state space, i.e. all possible MarkovState(s)
        transition_matrix -- MarkovTransitionMatrix object representing the transition matrix and its functions
        total_steps -- int representing the number of states we'll calculate past the initial time_step
    """

    def __init__(self, initial_state, state_space, transition_matrix, total_steps):
        self.initial_state = initial_state  # reminder: this is a MarkovStateVector object
        self.state_space = state_space  # reminder: this is a MarkovStateSpace object
        self.transition_matrix = transition_matrix  # reminder: this is a MarkovTransitionMatrix object
        self.total_steps = total_steps

        # do a few value checks on the sizes of the states and matrix
        if self.state_space.size != len(self.initial_state.state_distribution):
            raise ValueError(
                'MarkovChain.state_space must be the same size as MarkovChain.initial_state.state_distribution'
            )
        if self.state_space.size != self.transition_matrix.transition_function_matrix.shape[0]:
            raise ValueError(
                '''MarkovChain.transition_matrix must be a square matrix 
                with each dimension the size of MarkovChain.state_space'''
            )

    def __repr__(self):
        return 'MarkovChain(initial_state={}, state_space={})'.format(self.initial_state, self.state_space)

    def next_state(self, starting_state):
        """return a MarkovStateVector object after applying the transition matrix"""
        next_state = copy.deepcopy(starting_state)  # make a copy of the starting MarkovStateVector so we can update it
        next_state.time_step += 1  # add 1 to the starting time_step

        # grab the starting state distribution and take the dot product of the transition matrix at the time_step
        next_state.state_distribution = starting_state.state_distribution.dot(
            self.transition_matrix.matrix_at_time_step(starting_state.time_step)
        )

        return next_state

    def non_vectorized_state_after_n_steps(self, starting_state, n):
        """return a MarkovStateVector object after applying n transitions"""
        current_state = copy.deepcopy(starting_state)

        for step in np.arange(n):
            current_state = self.next_state(current_state)

        return current_state

    def state_after_n_steps(self, starting_state, n):
        """return a MarkovStateVector object after applying n transitions

        vectorize to allow for array inputs for either starting_state or n, not both
        """
        vectorized_func = np.vectorize(self.non_vectorized_state_after_n_steps)
        return vectorized_func(starting_state, n)
