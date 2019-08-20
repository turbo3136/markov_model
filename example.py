"""Example implementation of the markov_model package

Example:
    Let's imagine we have a weather system with three states, clear, windy, and rainy. The transition probability
    to a different state is an exponential decay, meaning the state is likely to stay the same over time. We fit the
    transition functions to the data provided and then observe the output.
"""

import time
start = time.time()

from datetime import datetime
import numpy as np
from markov_model.MarkovModel import MarkovModel

# debugging only
import pandas as pd
pd.set_option('display.expand_frame_repr', False)


def exp_decay(t, size):
    return size * np.exp(-t)


def flat_line(t, height):
    return height


total_steps_test = 4

transitions_dict = {
    'cohort': [
        '2019-01-01',
        '2019-01-01',
        '2019-01-01',
        '2019-01-01',
        '2019-01-01',
        '2019-01-01',
        '2019-01-01',
        '2019-01-01',
        '2019-01-01',
    ],
    'old_state_id': [
        'clear',
        'clear',
        'clear',
        'windy',
        'windy',
        'windy',
        'rainy',
        'rainy',
        'rainy',
    ],
    'new_state_id': [
        'clear',
        'windy',
        'rainy',
        'clear',
        'windy',
        'rainy',
        'clear',
        'windy',
        'rainy',
    ],
    'transition_function': [
        flat_line,
        exp_decay,
        exp_decay,
        exp_decay,
        flat_line,
        exp_decay,
        exp_decay,
        exp_decay,
        flat_line,
    ],
    'args': [
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
    ],
    'transition_probability': [
        [0.6, 0.86, 0.94],
        [0.2, 0.07, 0.03],
        [0.2, 0.07, 0.03],
        [0.2, 0.07, 0.03],
        [0.6, 0.86, 0.94],
        [0.2, 0.07, 0.03],
        [0.2, 0.07, 0.03],
        [0.2, 0.07, 0.03],
        [0.6, 0.86, 0.94],
    ],
    'transition_sigma': [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ],
    'args_initial_guess': [
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
        [0.1],
    ],
    'args_bounds': [
        ([0], [1]),
        ([0], [1]),
        ([0], [1]),
        ([0], [1]),
        ([0], [1]),
        ([0], [1]),
        ([0], [1]),
        ([0], [1]),
        ([0], [1]),
    ],
    'allow_fit': [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
}

state_dict = {
    'cohort': ['2019-01-01', '2019-01-01', '2019-01-01'],
    'state_id': ['clear', 'windy', 'rainy'],
    'distribution': [1, 0, 0],
    'count': [100, 0, 0],
    'time_step': [3, 3, 3],
}

state_df_test = pd.DataFrame.from_dict(state_dict)
transitions_df_test = pd.DataFrame.from_dict(transitions_dict)

mm = MarkovModel(initial_state_df=state_df_test, transitions_df=transitions_df_test, total_steps=total_steps_test, time_step_interval='day')

for cohort, chain in mm.markov_chains.items():
    print(chain.state_distribution_history())
    # print(chain.history)
    # for vector in chain.history:
    #     print(vector.time_step, vector.state_distribution, vector.current_date)
    # print(chain.markov_transition_matrix.matrix_at_time_step(3))
    # print(sum(chain.current_state.state_distribution))

end = time.time()
print(end - start)
