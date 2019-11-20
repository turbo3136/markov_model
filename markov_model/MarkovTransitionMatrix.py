import warnings
import pandas as pd
from markov_model.MarkovTransitionFunction import MarkovTransitionFunction


class MarkovTransitionMatrix:
    """ DON'T FORGET YOUR DOCSTRING!

    You forgot your docstring, didn't you...
    """

    def __init__(
            self,
            transition_matrix_df
    ):
        self.transition_matrix_df = transition_matrix_df
        self.column_names = self.transition_matrix_df.columns.values

        self.total_column = 'row_total'

    def __repr__(self):
        return '< MarkovTransitionMatrix object ¯\\_(ツ)_/¯ >'

    def matrix_at_time_step(self, time_step):
        # first, we create a temp DF that returns a tuple (element 0, element 1)
        # element 0: calculates the value of each TF at the time step (is_remainder TF returns 0)
        # element 1: TF.is_remainder (useful to figure out if this cell needs to be recalculated)
        ret = self.transition_matrix_df.applymap(lambda x: (x.value_at_time_step(time_step), x.is_remainder))

        # second, we create a row_total column that sums up each element of the tuples
        # element 0: the row's total transition probability
        # element 1: the row's total value for is_remainder
        ret[self.total_column] = ret.apply(self.sum_elements_of_tuples, axis=1)

        # third, we fill in the remainder where needed
        ret[self.column_names] = ret.apply(self.use_remainder_to_create_row, axis=1)

        return ret[self.column_names]

    @staticmethod
    def sum_elements_of_tuples(row):
        ret = tuple(map(sum, zip(*row)))

        # if the second element is > 1, throw an error. This means two TFs are the remainder. IT CAN'T BE!
        if ret[1] > 1:
            raise ValueError(
                'transition matrix row {} contains more than 1 cell claiming to be the remainder'.format(row.name)
            )

        # if the first element is not in [0,1], throw an error. Transition probabilities must be between 0 and 1.
        if ret[0] < 0 or ret[0] > 1:
            raise ValueError('transition matrix row {} is out of bounds. row total = {}'.format(row.name, ret[0]))

        return ret

    def use_remainder_to_create_row(self, row):
        # if the second element is not truthy, return the value, otherwise return the remainder
        ret = [tup[0] if not tup[1] else 1 - row[self.total_column][0] for tup in row[self.column_names]]

        # if the sum of the row is not 1, throw an error. Transition matrices must have rows sum to 1.
        if round(sum(ret), 6) != 1:
            raise ValueError('transition matrix row {} does not sum to 1. row total = {}'.format(row.name, sum(ret)))

        return pd.Series(ret)
