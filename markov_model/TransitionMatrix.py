import numpy as np


class TransitionMatrix:
    """Create a TransitionMatrix object to operate on a MarkovStateVector object

    Keyword arguments:
        state_space -- MarkovStateSpace object for this system, i.e. a list of all possible MarkovState(s)
        transition_function_list -- list of TransitionFunction objects, one for each combination in our state space
    """

    def __init__(self, state_space, transition_function_list):
        self.state_space = state_space
        self.transition_function_list = transition_function_list

        # now we want to create the state pair matrix from the state space object
        self.state_array = self.state_space.state_array  # first we need to get the array for the state space
        self.state_pair_matrix = [[(i, j) for j in self.state_array] for i in self.state_array]

        # now that we have a matrix of states, we want to create a matrix of tuple ids representing the transitions
        self.state_id_tuple_matrix = [[(i.state_id, j.state_id) for j in self.state_array] for i in self.state_array]
        self.transition_function_matrix = [
            [
                tf for tf in self.transition_function_list for tup in row if tup == tf.state_id_tuple
            ] for row in self.state_id_tuple_matrix
        ]

    def __repr__(self):
        return 'TransitionMatrix(state_space={}, transition_function_list={})'.format(
            self.state_space, self.transition_function_list
        )

    def matrix_at_time_step(self, time_step):
        """"returns numpy matrix"""
        return np.matrix(list(
            map(
                lambda x: list(map(lambda y: y.value_at_time_step(time_step=time_step), x)),  # function we're mapping
                self.transition_function_matrix  # matrix we're mapping over. Notice the nested maps
            )
        ))


if __name__ == '__main__':
    from markov_model.MarkovState import MarkovState
    from markov_model.MarkovStateSpace import MarkovStateSpace
    from markov_model.TransitionFunction import TransitionFunction

    def one_over_t(t):
        if t != 0:
            return 1/t
        else:
            return 1

    s1 = MarkovState(state_id='s1')
    s2 = MarkovState(state_id='s2')
    sa = np.array([s1, s2])
    ss = MarkovStateSpace(state_array=sa)
    tf_11 = TransitionFunction(state_id_tuple=('s1', 's1'), transition_function=one_over_t)
    tf_12 = TransitionFunction(state_id_tuple=('s1', 's2'), transition_function=one_over_t)
    tf_21 = TransitionFunction(state_id_tuple=('s2', 's1'), transition_function=one_over_t)
    tf_22 = TransitionFunction(state_id_tuple=('s2', 's2'), transition_function=one_over_t)
    tfl = [tf_11, tf_12, tf_21, tf_22]

    tm = TransitionMatrix(state_space=ss, transition_function_list=tfl)

    print(tm.state_id_tuple_matrix)
    print(tm.transition_function_matrix)
    # print(np.matrix(tm.transition_function_matrix))
    print(tm.matrix_at_time_step(time_step=2))
