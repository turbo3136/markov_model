from markov_model.MarkovChain import MarkovChain


class MarkovModel:
    """ DON'T FORGET YOUR DOCSTRING!

    You forgot your docstring, didn't you...
    """
    def __init__(
            self,
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
            allow_fit_column=True,
            self_is_remainder=True,
            markov_transition_function_column='markov_transition_function_column',
    ):
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

        # first, we get a list of all unique cohorts and make sure the cohorts are the same across inputs
        self.unique_cohorts = self.initial_state_df[self.cohort_column].unique()
        if set(self.unique_cohorts) != set(self.transitions_df[self.cohort_column].unique()):
            raise ValueError('unique cohorts are different in initial_state_df and transition_df')

        # then we create the markov chains for each cohort
        self.markov_chains = self.create_markov_chains()

    def create_markov_chains(self):
        """for each cohort, return a MarkovChain object

        :return: dictionary of MarkovChain objects where the key is the cohort and the value is the object
        """
        return {
            cohort: MarkovChain(
                cohort=cohort,
                initial_state_df=self.initial_state_df.loc[self.initial_state_df[self.cohort_column] == cohort],
                transitions_df=self.transitions_df.loc[self.transitions_df[self.cohort_column] == cohort],
                total_steps=self.total_steps,

                initial_state_column=self.initial_state_column,
                initial_state_distribution_column=self.initial_state_distribution_column,
                initial_state_time_step_column=self.initial_state_time_step_column,

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
            ) for cohort in self.unique_cohorts
        }


if __name__ == '__main__':
    from datetime import datetime
    import numpy as np

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
            datetime(2019, 1, 1),
            datetime(2019, 1, 1),
            datetime(2019, 1, 1),
            datetime(2019, 1, 1),
            datetime(2019, 1, 1),
            datetime(2019, 1, 1),
            datetime(2019, 1, 1),
            datetime(2019, 1, 1),
            datetime(2019, 1, 1),
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
        'cohort': [datetime(2019, 1, 1), datetime(2019, 1, 1), datetime(2019, 1, 1)],
        'state_id': ['clear', 'windy', 'rainy'],
        'distribution': [100, 0, 0],
        'time_step': [3, 3, 3],
    }

    state_df_test = pd.DataFrame.from_dict(state_dict)
    transitions_df_test = pd.DataFrame.from_dict(transitions_dict)
    # print(state_df_test.head())
    # print(transitions_df_test.head())

    mm = MarkovModel(initial_state_df=state_df_test, transitions_df=transitions_df_test, total_steps=total_steps_test)

    for cohort, chain in mm.markov_chains.items():
        print(chain.history)
        print(chain.markov_transition_matrix.matrix_at_time_step(3))
        print(sum(chain.current_state.state_distribution))
