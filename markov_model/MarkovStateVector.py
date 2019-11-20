import numpy as np
import helpers


class MarkovStateVector:
    """Create a MarkovStateVector object defining a probability distribution within the state space.

    Keyword arguments:
        cohort -- identifier for the cohort, usually a datetime object
        state_space -- MarkovStateSpace object for this system, i.e. a list of all possible MarkovState(s)
        state_distribution_df -- pandas DataFrame representing the probability distribution within the state space
            , index: (cohort, state_id), columns: distribution
        markov_transition_matrix_df -- pandas DataFrame representing the transition matrix at time_step
        state_transition_df -- pandas DataFrame representing the transition probabilities between states
        cohort_size -- size of the state vector, or the total count of things in the state_distribution
        time_step -- time step for the system this vector represents
        time_step_interval -- the interval represented by each time step ('year', 'month', 'day')
    """

    def __init__(
            self,
            cohort,
            state_space,
            state_distribution_df,
            cohort_column,
            old_state_id_column,
            time_step_column,
            date_column,
            distribution_column,
            count_column,
            markov_transition_matrix_df,
            state_transition_df,
            cohort_size,
            time_step,
            time_step_interval,
    ):
        self.cohort = cohort
        self.state_space = state_space
        self.state_distribution_df = state_distribution_df
        self.cohort_column = cohort_column
        self.old_state_id_column = old_state_id_column
        self.time_step_column = time_step_column
        self.date_column = date_column
        self.distribution_column = distribution_column
        self.count_column = count_column
        self.markov_transition_matrix_df = markov_transition_matrix_df
        self.state_transition_df = state_transition_df
        self.cohort_size = cohort_size
        self.time_step = time_step
        self.time_step_interval = time_step_interval

        self.initial_time_step = time_step

        self.current_date = helpers.add_interval_to_date(
            date_object=helpers.date_string_to_datetime(self.cohort),
            steps=self.time_step,
            interval=self.time_step_interval,
        )

        if self.state_space.size != self.state_distribution_df[self.distribution_column].count():
            raise ValueError(
                'MarkovStateVector.state_distribution_df must be the same size as state_space array'
            )

        self.next_state_distribution_df = self.calc_next_state_distribution_df()

    def __repr__(self):
        return 'MarkovStateVector(cohort={}, state_space={}, time_step={})'.format(
            self.cohort, self.state_space, self.time_step
        )

    def calc_next_state_distribution_df(self):
        # first we do a few things:
        #   1. sum all the states to get the new distribution by state (we have a wide DF now)
        #   2. reset the index to prepare for the melt (we want to preserve the index while pivoting to a long DF)
        #       see --> https://stackoverflow.com/questions/50529022/pandas-melt-unmelt-preserve-index
        #   3. do the melt, which pivots the state columns into rows
        ret = self.state_transition_df.groupby(
            level=[self.cohort_column]
        ).sum().reset_index().melt(
            id_vars=self.cohort_column,
            var_name=self.old_state_id_column,
            value_name=self.distribution_column,
        )

        # create the new time_step column
        ret[self.time_step_column] = self.time_step + 1

        # create the new date column
        ret[self.date_column] = helpers.add_interval_to_date(
            date_object=helpers.date_string_to_datetime(self.cohort),
            steps=self.time_step + 1,
            interval=self.time_step_interval,
        )

        # create the count column
        ret[self.count_column] = ret[self.distribution_column] * self.cohort_size

        # set the index and return
        return ret.set_index(
            [self.cohort_column, self.date_column, self.time_step_column, self.old_state_id_column]
        )
