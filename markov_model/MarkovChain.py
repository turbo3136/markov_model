import numpy as np
import pandas as pd
import helpers
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

            fit_data=False,
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

            markov_transition_function_column='markov_transition_function_column',
            time_step_interval='month',
            date_column='date'
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

        self.markov_transition_function_column = markov_transition_function_column
        self.time_step_interval = time_step_interval
        self.date_column = date_column

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

        # then we create the transition matrix df by cohort (index = (cohort, old_state_id), columns = new_state_id)
        self.transition_matrix_df = pd.pivot_table(
            self.transitions_df,
            values=self.markov_transition_function_column,
            index=[self.cohort_column, self.old_state_id_column],
            columns=[self.new_state_id_column],
            aggfunc=lambda x: x,  # we don't actually want to aggregate anything, we're just exploiting the pivot table
        )

        # then we create the MarkovTransitionMatrix
        self.markov_transition_matrix = MarkovTransitionMatrix(transition_matrix_df=self.transition_matrix_df)

        # now we figure out the size of the initial state (i.e. the total number of things in all the states)
        self.cohort_size = self.initial_state_df[self.initial_state_count_column].sum()

        # now we grab the time step
        self.time_step_set = set(self.initial_state_df[self.initial_state_time_step_column].values)
        if len(self.time_step_set) != 1:
            raise ValueError(
                'the initial time step has multiple values for the same cohort, cohort={}'.format(self.cohort)
            )
        self.initial_state_time_step = self.time_step_set.pop()

        # and the initial date
        self.initial_state_date = helpers.add_interval_to_date(
            date_object=helpers.date_string_to_datetime(self.cohort),
            steps=self.initial_state_time_step,
            interval=self.time_step_interval,
        )

        # and then we create the state_distribution_df (index=(cohort, state_id), column=distribution)
        self.state_distribution_df = self.initial_state_df.rename(
            columns={self.initial_state_column: self.old_state_id_column}  # rename the state_id column to old_state_id
        )

        self.state_distribution_df[self.date_column] = self.initial_state_date

        self.state_distribution_df = self.state_distribution_df.set_index(
            # set the index to cohort, date, time_step, old_state_id
            [self.cohort_column, self.date_column, self.initial_state_time_step_column, self.old_state_id_column]
        )[[self.initial_state_distribution_column, self.initial_state_count_column]]

        # grab the matrix at the initial time step
        self.initial_markov_transition_matrix = self.markov_transition_matrix.matrix_at_time_step(
            self.initial_state_time_step
        )

        # and create the state_transition_df for the initial time_step
        self.state_transition_df = helpers.join_vector_to_df_on_index_and_multiply_across_rows(
            self.state_distribution_df, self.initial_markov_transition_matrix, self.initial_state_distribution_column
        )

        # finally, let's create the MarkovStateVector object
        self.markov_state_vector = MarkovStateVector(
            cohort=self.cohort,
            state_space=self.markov_state_space,
            state_distribution_df=self.state_distribution_df,
            cohort_column=self.cohort_column,
            old_state_id_column=self.old_state_id_column,
            time_step_column=self.initial_state_time_step_column,
            date_column=self.date_column,
            distribution_column=self.initial_state_distribution_column,
            count_column=self.initial_state_count_column,
            markov_transition_matrix_df=self.initial_markov_transition_matrix,
            state_transition_df=self.state_transition_df,
            cohort_size=self.cohort_size,
            time_step=self.initial_state_time_step,
            time_step_interval=self.time_step_interval
        )

        self.history = [self.markov_state_vector]  # initialize the history with the initial state

        # then we have everything we need to calculate the current state and log the history along the way
        self.current_state = self.state_after_n_steps(
            self.markov_state_vector, self.total_steps, log_history=True
        )

        self.state_distribution_history_df = self.state_distribution_history()
        self.state_transition_history_df = self.state_transition_history()

    def __repr__(self):
        return 'MarkovChain(current_state={}, state_space={})'.format(
            self.current_state, self.markov_state_space
        )

    def next_state(self, starting_state, log_history=False):
        """return a MarkovStateVector object after applying the transition matrix"""
        next_state_distribution_df = starting_state.next_state_distribution_df

        next_markov_transition_matrix_df = self.markov_transition_matrix.matrix_at_time_step(
            starting_state.time_step + 1
        )
        next_state_transition_df = helpers.join_vector_to_df_on_index_and_multiply_across_rows(
            next_state_distribution_df, next_markov_transition_matrix_df, starting_state.distribution_column
        )

        next_state = MarkovStateVector(
            cohort=starting_state.cohort,
            state_space=starting_state.state_space,
            state_distribution_df=next_state_distribution_df,
            cohort_column=starting_state.cohort_column,
            old_state_id_column=starting_state.old_state_id_column,
            time_step_column=starting_state.time_step_column,
            date_column=starting_state.date_column,
            distribution_column=starting_state.distribution_column,
            count_column=starting_state.count_column,
            markov_transition_matrix_df=next_markov_transition_matrix_df,
            state_transition_df=next_state_transition_df,
            cohort_size=starting_state.cohort_size,
            time_step=starting_state.time_step + 1,  # increment the time_step
            time_step_interval=starting_state.time_step_interval,
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
            state_distribution_df=starting_state.state_distribution_df,
            cohort_column=starting_state.cohort_column,
            old_state_id_column=starting_state.old_state_id_column,
            time_step_column=starting_state.time_step_column,
            date_column=starting_state.date_column,
            distribution_column=starting_state.distribution_column,
            count_column=starting_state.count_column,
            markov_transition_matrix_df=starting_state.markov_transition_matrix_df,
            state_transition_df=starting_state.state_transition_df,
            cohort_size=starting_state.cohort_size,
            time_step=starting_state.time_step,
            time_step_interval=starting_state.time_step_interval,
        )

        for step in np.arange(n):
            current_state = self.next_state(current_state, log_history)

        return current_state

    def state_distribution_history(self):
        """dataframe of state distribution with current date, time_step, and state_ids as the columns

        output looks like:
            cohort      date        time_step   state_id      distribution  count
            2019-01-01  2019-01-01  0           state_i       0.4           88
            2019-01-01  2019-01-01  0           state_j       0.4           88
            2019-01-01  2019-01-01  0           state_k       0.2           44
            .
            .
            .
        """
        state_distribution_df_list = [v.state_distribution_df for v in self.history]
        return pd.concat(state_distribution_df_list)

    # TODO: dataframe of transition probability and counts for state pairs (state_i, state_j) by current date
    def state_transition_history(
            self,
            transition_probability_column='transition_probability',
            transition_count_column='transition_count',
    ):
        """dataframe of transition probability between state_id tuples

        output looks like:
            cohort      date        time_step   old_state_id    new_state_id    transition_probability  transition_count
            2019-01-01  2019-01-01  0           state_i         state_i         0.4                     88
            2019-01-01  2019-01-01  0           state_i         state_j         0.4                     88
            2019-01-01  2019-01-01  0           state_i         state_k         0.2                     44
            .
            .
            .
        """
        state_transition_df_list = [
            self.melt_state_transition_df_and_transition_matrix_and_join(
                v, transition_probability_column, transition_count_column
            )
            for v in self.history
        ]
        return pd.concat(state_transition_df_list)

    def melt_state_transition_df_and_transition_matrix_and_join(
            self,
            markov_state,
            transition_probability_column,
            transition_count_column,
    ):
        ret = markov_state.state_transition_df * markov_state.cohort_size
        ret = ret.reset_index().melt(
            id_vars=[
                self.cohort_column, self.initial_state_time_step_column, self.date_column, self.old_state_id_column
            ],
            var_name=self.new_state_id_column,
            value_name=transition_count_column,
        )

        tm = markov_state.markov_transition_matrix_df.reset_index().melt(
            id_vars=[self.cohort_column, self.old_state_id_column],
            var_name=self.new_state_id_column,
            value_name=transition_probability_column
        )

        ret = ret.merge(tm, on=[self.cohort_column, self.old_state_id_column, self.new_state_id_column])

        return ret.set_index([
            self.cohort_column,
            self.date_column,
            self.initial_state_time_step_column,
            self.old_state_id_column,
            self.new_state_id_column,
        ])[[transition_probability_column, transition_count_column]]
