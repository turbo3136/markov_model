class TransitionFunction:
    """Create a TransitionFunction object defining the probability of transitioning between two MarkovStates

        Keyword arguments:
            state_id_tuple -- unique identifier, tuple containing the id of the initial state and the end state
            transition_function -- time dependent function representing the transition probability between
                initial and end states. transition_function must take exactly one argument, t (time_step)
        """

    def __init__(self, state_id_tuple, transition_function):
        self.state_id_tuple = state_id_tuple
        self.transition_function = transition_function

    def __repr__(self):
        return 'TransitionFunction(state_id_tuple={})'.format(self.state_id_tuple)

    def value_at_time_step(self, time_step):
        return self.transition_function(time_step)


if __name__ == '__main__':
    def one_over_t(t):
        if t != 0:
            return 1/t
        else:
            return 1

    tf = TransitionFunction(state_id_tuple=('hello', 'world'), transition_function=one_over_t)
    print(tf.value_at_time_step(time_step=4))
