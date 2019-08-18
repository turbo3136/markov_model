import numpy as np
import copy
from markov_model.MarkovStateSpace import MarkovStateSpace
from markov_model.MarkovStateVector import MarkovStateVector
from markov_model.MarkovTransitionMatrix import MarkovTransitionMatrix


class MarkovChain:
    """Create a MarkovChain object based on an initial state, state space, and transition matrix

    Keyword arguments:


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

    def __init__(
            self,
            cohort,
            initial_state_df,
            transitions_df,
            total_steps,

            initial_state_column='state_id',
            initial_state_distribution_column='distribution',
            initial_state_time_step_column='time_step',

            fit_data=True,
            cohort_column='cohort',
            old_state_id_column='old_state_id',
            new_state_id_column='new_state_id',
            transition_function_column='transition_function',
            args_column='args',
            xdata_column=None,
            ydata_column='transition_probability',
            ydata_sigma_column='transition_sigma',
            args_initial_guess_column='args_initial_guess',
            args_bounds_column='args_bounds',
            allow_fit_column='allow_fit',
            self_is_remainder=True,
            markov_transition_function_column='markov_transition_function_column',
    ):
        self.cohort = cohort
        self.initial_state_df = initial_state_df
        self.transitions_df = transitions_df
        self.total_steps = total_steps

        self.initial_state_column = initial_state_column
        self.initial_state_distribution_column = initial_state_distribution_column
        self.initial_state_time_step_column = initial_state_time_step_column

        self.fit_data = fit_data
        self.cohort_column = cohort_column
        self.old_state_id_column = old_state_id_column
        self.new_state_id_column = new_state_id_column
        self.transition_function_column = transition_function_column
        self.args_column = args_column
        self.xdata_column = xdata_column
        self.ydata_column = ydata_column
        self.ydata_sigma_column = ydata_sigma_column
        self.args_initial_guess_column = args_initial_guess_column
        self.args_bounds_column = args_bounds_column
        self.allow_fit_column = allow_fit_column
        self.self_is_remainder = self_is_remainder
        self.markov_transition_function_column = markov_transition_function_column

        # check to see if we have more than one cohort, if so raise an error
        if len(self.initial_state_df[self.cohort_column].unique()) != 1 or \
                len(self.transitions_df[self.cohort_column].unique()) != 1:
            raise ValueError('MarkovChain object passed dataframe with more than one unique cohort')

        # now let's check to make sure the states spaces are the same for each of the inputs
        self.state_id_list = self.initial_state_df[self.initial_state_column].unique()
        if set(self.state_id_list) != set(self.transitions_df[self.old_state_id_column].unique()) or \
                set(self.state_id_list) != set(self.transitions_df[self.new_state_id_column].unique()):
            raise ValueError(
                'unique states in initial_state_df and transitions_df for cohort={} not equal to each other'
                .format(self.cohort)
            )

        # now we create the MarkovStateSpace object
        self.markov_state_space = MarkovStateSpace(state_id_list=self.state_id_list)

        # now let's create the initial_state_dict
        self.initial_state_dict = self.initial_state_df[
            [self.initial_state_column, self.initial_state_distribution_column, self.initial_state_time_step_column]
        ].to_dict(orient='list')
        # let's check to make sure the distribution is in the same order as we expect from MarkovStateSpace
        if not all(self.state_id_list == self.initial_state_dict[self.initial_state_column]):
            raise ValueError('looks like the state distribution got out of order, fuck')

        # now we create the initial state distribution and add it to the history log
        self.initial_state_distribution = np.array(self.initial_state_dict[self.initial_state_distribution_column])

        # now let's grab the initial time_step
        if len(set(self.initial_state_dict[self.initial_state_time_step_column])) != 1:
            raise ValueError(
                'the initial time step has multiple values for the same cohort, cohort='.format(self.cohort)
            )
        self.initial_state_time_step = self.initial_state_dict[self.initial_state_time_step_column][0]

        # now let's create the MarkovStateVector object
        self.markov_state_vector = MarkovStateVector(
            state_space=self.markov_state_space,
            state_distribution=self.initial_state_distribution,
            time_step=self.initial_state_time_step,
        )
        self.history = np.array([self.markov_state_vector])  # initialize the history with the initial state

        # now we create the MarkovTransitionMatrix object
        self.markov_transition_matrix = MarkovTransitionMatrix(
            cohort=self.cohort,
            state_space=self.markov_state_space,
            transitions_df=self.transitions_df,

            fit_data=self.fit_data,
            cohort_column=self.cohort_column,
            old_state_id_column=self.old_state_id_column,
            new_state_id_column=self.new_state_id_column,
            transition_function_column=self.transition_function_column,
            args_column=self.args_column,
            xdata_column=self.xdata_column,
            ydata_column=self.ydata_column,
            ydata_sigma_column=self.ydata_sigma_column,
            args_initial_guess_column=self.args_initial_guess_column,
            args_bounds_column=self.args_bounds_column,
            allow_fit_column=self.allow_fit_column,
            self_is_remainder=self.self_is_remainder,
            markov_transition_function_column=self.markov_transition_function_column,
        )

        # then we have everything we need to calculate the current state and log the history along the way
        self.current_state = self.state_after_n_steps(
            self.markov_state_vector, self.total_steps, log_history=True
        )

    def __repr__(self):
        return 'MarkovChain(current_state={}, state_space={})'.format(
            self.current_state, self.markov_state_space
        )

    def next_state(self, starting_state, log_history=False, make_deepcopy=True):
        """return a MarkovStateVector object after applying the transition matrix"""
        if make_deepcopy:  # make a copy of the starting MarkovStateVector so we can update it
            next_state = copy.deepcopy(starting_state)
        else:  # otherwise, just add a new reference
            next_state = starting_state

        next_state.time_step += 1  # add 1 to the starting time_step

        # grab the starting state distribution and take the dot product of the transition matrix at the time_step
        next_state.state_distribution = starting_state.state_distribution.dot(
            self.markov_transition_matrix.matrix_at_time_step(starting_state.time_step)
        )

        # check to see if we want to log the history
        if log_history:
            self.history = np.append(self.history, next_state)  # add the next state to the history log

        return next_state

    def state_after_n_steps(self, starting_state, n, log_history=False):
        """return a MarkovStateVector object after applying n transitions"""
        current_state = copy.deepcopy(starting_state)

        for step in np.arange(n):
            current_state = self.next_state(current_state, log_history)

        return current_state

    def vectorized_state_after_n_steps(self, starting_state, n, log_history=False):
        """return a MarkovStateVector object after applying n transitions

        vectorize to allow for array inputs for either starting_state or n, not both
        """
        vectorized_func = np.vectorize(self.state_after_n_steps)
        return vectorized_func(starting_state, n, log_history)
