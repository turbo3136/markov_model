"""Example implementation of the markov_model package

Example:
    Let's imagine we have a system with two states. State 1 (s1) has time dependent transition probabilities
    and State 2 (s2) has fixed transition probabilities of 0.5. Below, we create the necessary objects and
    print out the state after n steps.
"""

import time
start = time.time()

import numpy as np
from markov_model.MarkovState import MarkovState
from markov_model.MarkovStateSpace import MarkovStateSpace
from markov_model.MarkovStateVector import MarkovStateVector
from markov_model.MarkovTransitionFunction import MarkovTransitionFunction
from markov_model.MarkovTransitionMatrix import MarkovTransitionMatrix
from markov_model.MarkovChain import MarkovChain


# transition functions
def exp_decay(t):
    return np.exp(-t)


def one_minus_exp_decay(t):
    return 1 - exp_decay(t)


def point_five(t):
    return 0.5


# states
s1 = MarkovState(state_id='s1')
s2 = MarkovState(state_id='s2')
sa = np.array([s1, s2])  # state_array
# state space
ss = MarkovStateSpace(state_array=sa)
# transition functions
tf_11 = MarkovTransitionFunction(state_id_tuple=('s1', 's1'), transition_function=exp_decay)
tf_12 = MarkovTransitionFunction(state_id_tuple=('s1', 's2'), transition_function=one_minus_exp_decay)
tf_21 = MarkovTransitionFunction(state_id_tuple=('s2', 's1'), transition_function=point_five)
tf_22 = MarkovTransitionFunction(state_id_tuple=('s2', 's2'), transition_function=point_five)
tfl = [tf_11, tf_12, tf_21, tf_22]
# transition matrix
tm = MarkovTransitionMatrix(state_space=ss, transition_function_list=tfl)
# state vector
sd = np.array([1, 0])  # state_distribution
ts = 0  # time_step
sv = MarkovStateVector(state_space=ss, state_distribution=sd, time_step=ts)

# markov chain
mc = MarkovChain(initial_state=sv, state_space=ss, cohort='test', transition_matrix=tm, total_steps=10)

print(mc.initial_state)
# print(mc.transition_matrix.matrix_at_time_step(time_step=0))
# print(mc.state_after_n_steps(mc.initial_state, 10))
# print(mc.vectorized_state_after_n_steps(mc.initial_state, np.arange(100)))  # vectorization!
# print(mc.history)
print(mc.current_state)

end = time.time()
print(end - start)
