import numpy as np
from markov_model.MarkovTransitionFunction import MarkovTransitionFunction


class MarkovTransitionMatrix:
    """Create a MarkovTransitionMatrix object to operate on a MarkovStateVector object
    Keyword arguments:
        cohort -- identifier for the cohort, usually a datetime object
        state_space -- MarkovStateSpace object for this system, i.e. a list of all possible MarkovState(s)
        transitions_df
        fit_data
        cohort_column
        old_state_id_column
        new_state_id_column
        transition_function_column
        args_column
        xdata_column
        ydata_column
        ydata_sigma_column
        args_initial_guess_column
        args_bounds
        allow_fit
        self_is_remainder
    """

    def __init__(
            self,
            cohort,
            state_space,
            transitions_df,

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
            markov_transition_function_column='markov_transition_function',
    ):
        self.cohort = cohort
        self.state_space = state_space
        self.transitions_df = transitions_df

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

        # add the markov_transition_function column to the dataframe, made of MarkovTransitionFunction objects
        self.transitions_df[self.markov_transition_function_column] = self.transitions_df.apply(
            self.create_markov_transition_function_column, axis=1
        )
        # and then create the list of MarkovTransitionFunctions
        self.transition_function_list = self.transitions_df[self.markov_transition_function_column].tolist()

        # now we want to create the state pair matrix from the state space object
        self.state_array = self.state_space.state_array  # first we need to get the array for the state space
        self.state_pair_matrix = [[(i, j) for j in self.state_array] for i in self.state_array]

        # now that we have a matrix of states, we want to create a matrix of tuple ids representing the transitions
        self.state_id_tuple_matrix = [[(i.state_id, j.state_id) for j in self.state_array] for i in self.state_array]
        self.state_id_tuple_list = [(i.state_id, j.state_id) for j in self.state_array for i in self.state_array]

        # now let's create a dictionary to lookup the transition_function for a state_id tuple
        self.transition_function_lookup = {
            tup: tf
            for tf in self.transition_function_list
            for tup in self.state_id_tuple_list
            if tup == tf.state_id_tuple
        }

        # let's create a dictionary to lookup the position of a state_id tuple within the matrix
        self.state_id_tuple_position_lookup = {
            tup: (i, j)
            for i, row in enumerate(self.state_id_tuple_matrix)
            for j, tup in enumerate(row)
        }

        # now we create a numpy matrix of transition functions
        self.transition_function_matrix = np.array([
            [
                self.transition_function_lookup[tup] for tup in row
            ] for row in self.state_id_tuple_matrix
        ])

    def __repr__(self):
        return 'MarkovTransitionMatrix(state_space={}, transition_function_list={})'.format(
            self.state_space, self.transition_function_list
        )

    def matrix_at_time_step(self, time_step):
        ret = np.empty_like(self.transition_function_matrix)

        remainder_check = False  # create a check to see if any of the functions are supposed to be the remainder

        for row_index, row in enumerate(self.transition_function_matrix):
            for tf_index, tf in enumerate(row):
                if tf.is_remainder:
                    remainder_check = True
                    ret[row_index][tf_index] = 0
                else:
                    ret[row_index][tf_index] = tf.value_at_time_step(time_step)

        # TODO: this is not an elegant solution
        if remainder_check:  # if we want to set (state_i, state_i) transitions to the remainder, do it
            for row_index, row in enumerate(self.transition_function_matrix):
                for tf_index, tf in enumerate(row):
                    if tf.is_remainder:
                        ret[row_index][tf_index] = 1 - sum(ret[row_index])

        return ret

    def state_id_tuple_value_at_time_step(self, matrix_at_time_step, state_id_tuple):
        """take a matrix_at_time_step and return the transition probability for a state_id tuple"""
        position_tuple = self.state_id_tuple_position_lookup[state_id_tuple]

        return matrix_at_time_step[position_tuple[0], position_tuple[1]]

    def create_markov_transition_function_column(self, row):
        """take a row of a dataframe and return a MarkovTransitionFunction object"""
        ydata = row[self.ydata_column]  # first grab the ydata array
        if self.xdata_column is None:  # if we didn't provide xdata info, then create an array of length ydata
            xdata = np.arange(len(ydata))
        else:  # otherwise, grab the column provided
            xdata = row[self.xdata_column]

        # if (state_i, state_i) transitions should be categorized as the remainder, let's capture that info
        remainder_boolean = False
        if self.self_is_remainder and row[self.old_state_id_column] == row[self.new_state_id_column]:
            remainder_boolean = True

        ret = MarkovTransitionFunction(
            state_id_tuple=(row[self.old_state_id_column], row[self.new_state_id_column]),
            transition_function=row[self.transition_function_column],
            cohort=row[self.cohort_column],
            args=row[self.args_column],
            xdata=xdata,
            ydata=ydata,
            ydata_sigma=row[self.ydata_sigma_column],
            args_initial_guess=row[self.args_initial_guess_column],
            args_bounds=row[self.args_bounds_column],
            allow_fit=row[self.allow_fit_column],
            is_remainder=remainder_boolean,
        )

        if self.fit_data:
            ret.fit_to_data()

        return ret
