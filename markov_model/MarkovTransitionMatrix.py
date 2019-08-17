import numpy as np


class MarkovTransitionMatrix:
    """Create a MarkovTransitionMatrix object to operate on a MarkovStateVector object

    Keyword arguments:
        state_space -- MarkovStateSpace object for this system, i.e. a list of all possible MarkovState(s)
        transition_function_list -- list of MarkovTransitionFunction objects,
            one for each combination in our state space
        cohort -- optional, identifier for the cohort, if applicable
        self_is_remainder -- optional, boolean flag for setting the "self" transition to 1 - sum(other transitions),
            to be used if the (state_x, state_x) transition function is unknown, but we know the row must sum to 1
    """

    def __init__(self, state_space, transition_function_list, cohort=None):
        self.state_space = state_space
        self.transition_function_list = transition_function_list
        self.cohort = cohort

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

    # @staticmethod
    # def transition_function_value_at_time_step(transition_function, time_step):
    #     """just grab the value for the transition function at a certain time step"""
    #     return transition_function.value_at_time_step(time_step=time_step)

    # def matrix_at_time_step(self, time_step):
    #     """vectorize the value at time step function and apply it to the transition_function_matrix
    #
    #     returns numpy matrix
    #     """
    #     vectorized_func = np.vectorize(self.transition_function_value_at_time_step)
    #     return vectorized_func(self.transition_function_matrix, time_step)

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

        if remainder_check:  # if we want to set (state_i, state_i) transitions to the remainder, do it
            for row_index, row in enumerate(self.transition_function_matrix):
                for tf_index, tf in enumerate(row):
                    if tf.is_remainder:
                        ret[row_index][tf_index] = 1 - sum(ret[row_index])

        return ret


if __name__ == '__main__':
    from markov_model.MarkovState import MarkovState
    from markov_model.MarkovStateSpace import MarkovStateSpace
    from markov_model.MarkovTransitionFunction import MarkovTransitionFunction

    def one_over_t(t):
        if t != 0:
            return 1/t
        else:
            return 1

    s1 = MarkovState(state_id='s1')
    s2 = MarkovState(state_id='s2')
    sa = np.array([s1, s2])
    ss = MarkovStateSpace(state_array=sa)
    tf_11 = MarkovTransitionFunction(state_id_tuple=('s1', 's1'), transition_function=one_over_t, is_remainder=True)
    tf_12 = MarkovTransitionFunction(state_id_tuple=('s1', 's2'), transition_function=one_over_t)
    tf_21 = MarkovTransitionFunction(state_id_tuple=('s2', 's1'), transition_function=one_over_t, is_remainder=True)
    tf_22 = MarkovTransitionFunction(state_id_tuple=('s2', 's2'), transition_function=one_over_t)
    tfl = [tf_11, tf_12, tf_21, tf_22]

    tm = MarkovTransitionMatrix(state_space=ss, transition_function_list=tfl)

    print(tm.state_id_tuple_matrix)
    print(tm.transition_function_matrix)
    # print(np.matrix(tm.transition_function_matrix))
    print(tm.matrix_at_time_step(time_step=8))
    print(tm.transition_function_lookup)
    # print(tm.transition_function_matrix[0][0])
