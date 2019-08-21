import numpy as np
import pandas as pd
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
        next_state(starting_state, log_history) --
            return the state after starting_state
            if requested, log the history in self.history, default is False

        state_after_n_steps(starting_state, n, log_history) --
            return the state n steps after starting_state
            if requested, log the history in self.history, default is False
    """

    def __init__(
            self,
            cohort,
            initial_state_df,
            transitions_df,
            total_steps,

            initial_state_column='state_id',
            initial_state_distribution_column='distribution',
            initial_state_count_column='count',
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
            time_step_interval='month',
    ):
        self.cohort = cohort
        self.initial_state_df = initial_state_df
        self.transitions_df = transitions_df
        self.total_steps = total_steps

        self.initial_state_column = initial_state_column
        self.initial_state_distribution_column = initial_state_distribution_column
        self.initial_state_count_column = initial_state_count_column
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
        self.time_step_interval = time_step_interval

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

        # now let's sum up the initial state count column to figure out the total size of the vector
        self.cohort_size = self.initial_state_df[self.initial_state_count_column].sum()

        # now let's create the MarkovStateVector object
        self.markov_state_vector = MarkovStateVector(
            cohort=self.cohort,
            state_space=self.markov_state_space,
            state_distribution=self.initial_state_distribution,
            time_step=self.initial_state_time_step,
            time_step_interval=self.time_step_interval,
            size=self.cohort_size,
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

        # self.state_distribution_history()

    def __repr__(self):
        return 'MarkovChain(current_state={}, state_space={})'.format(
            self.current_state, self.markov_state_space
        )

    def next_state(self, starting_state, log_history=False):
        """return a MarkovStateVector object after applying the transition matrix"""
        mat = self.markov_transition_matrix.matrix_at_time_step(starting_state.time_step)

        next_state = MarkovStateVector(
            cohort=starting_state.cohort,
            state_space=starting_state.state_space,
            state_distribution=starting_state.state_distribution.dot(mat),  # increment the state_distribution
            time_step=starting_state.time_step + 1,  # increment the time_step
            time_step_interval=starting_state.time_step_interval,
            size=starting_state.size,
        )

        # check to see if we want to log the history
        if log_history:
            self.history = np.append(self.history, next_state)  # add the next state to the history log

        return next_state

    def state_after_n_steps(self, starting_state, n, log_history=False):
        """return a MarkovStateVector object after applying n transitions"""
        current_state = MarkovStateVector(
            cohort=starting_state.cohort,
            state_space=starting_state.state_space,
            state_distribution=starting_state.state_distribution,
            time_step=starting_state.time_step,
            time_step_interval=starting_state.time_step_interval,
            size=starting_state.size
        )

        for step in np.arange(n):
            current_state = self.next_state(current_state, log_history)

        return current_state

    def state_distribution_history(
            self,
            date_column='date',
            time_step_column='time_step',
            state_id_column='state_id',
            distribution_column='distribution',
            count_column='count',
    ):
        """dataframe of state distribution with current date, time_step, and state_ids as the columns

        output looks like:
            date        time_step   state_id      distribution  count
            2019-01-01  0           state_i       0.4           88
            2019-01-01  0           state_j       0.4           88
            2019-01-01  0           state_k       0.2           44
            .
            .
            .
        """
        ret = {date_column: [], time_step_column: [], state_id_column: [], distribution_column: [], count_column: []}
        for index, vector in enumerate(self.history):
            for state_id, distribution in vector.state_distribution_dict.items():
                ret[date_column].append(vector.current_date)  # add the date to the list
                ret[time_step_column].append(vector.time_step)  # add the time step to the list
                ret[state_id_column].append(state_id)  # add the state_id to the list
                ret[distribution_column].append(distribution)  # add the distribution in this state to the list
                ret[count_column].append(vector.size * distribution)  # add the count of items in this state

        return pd.DataFrame.from_dict(ret)

    # TODO: dataframe of transition probability and counts for state pairs (state_i, state_j) by current date
    def state_transition_history(
            self,
            date_column='date',
            time_step_column='time_step',
            old_state_id_column='old_state_id',
            new_state_id_column='new_state_id',
            transition_probability_column='transition_probability',
            transition_count_column='transition_count',
    ):
        """dataframe of transition probability between state_id tuples

        output looks like:
            date        time_step   old_state_id    new_state_id      transition_probability  transition_count
            2019-01-01  0           state_i         state_i           0.4                     88
            2019-01-01  0           state_i         state_j           0.4                     88
            2019-01-01  0           state_i         state_k           0.2                     44
            .
            .
            .
        """
        ret = {
            date_column: [],
            time_step_column: [],
            old_state_id_column: [],
            new_state_id_column: [],
            transition_probability_column: [],
            transition_count_column: [],
        }
        for index, vector in enumerate(self.history):
            # TODO: log the matrix history ahead of time so we don't have to calculate this again
            mat = self.markov_transition_matrix.matrix_at_time_step(time_step=vector.time_step)
            state_id_list = vector.state_space.state_id_list  # list of state_ids

            for i, row in enumerate(mat):
                for j, value in enumerate(row):
                    ret[date_column].append(vector.current_date)  # add the date to the list
                    ret[time_step_column].append(vector.time_step)  # add the time step to the list
                    ret[old_state_id_column].append(state_id_list[i])  # add the state_id to the list
                    ret[new_state_id_column].append(state_id_list[j])  # add the state_id to the list
                    ret[transition_probability_column].append(mat[i][j])  # add the transition_probability
                    # and finally the count of transitions between the old and new state
                    ret[transition_count_column].append(vector.state_distribution[i] * vector.size * mat[i][j])

        return pd.DataFrame.from_dict(ret)
