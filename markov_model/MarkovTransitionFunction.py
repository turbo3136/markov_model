import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objs as go
from plotly.offline import plot


class MarkovTransitionFunction:
    """Create a MarkovTransitionFunction object defining the probability of transitioning between two MarkovStates

    Keyword arguments:
        state_id_tuple -- unique identifier, tuple containing the id of the initial state and the end state
        transition_function -- time dependent function representing the transition probability between
            initial and end states. transition_function's first argument must be t (time_step),
            followed by optional args
        cohort -- optional, identifier for the cohort, if applicable
        args -- optional, values of optional arguments in transition function
        xdata -- optional, numpy array of x axis data to be used for curve_fit
        ydata -- optional, numpy array of y axis data to be used for curve_fit
        ydata_sigma -- optional, numpy array describing the variance of ydata
        y2daya -- optional, numpy array of secondary y axis data
        args_initial_guess -- optional, initial guess for args, used for fitting transition function to data
        args_bounds -- optional, 2-tuple of list, lower an upper bounds used for fitting transition function to data
        allow_fit -- optional, boolean for whether to allow fitting for this transition function
    """

    def __init__(
            self,
            state_id_tuple,
            transition_function,
            cohort=None,
            args=None,
            xdata=None,
            ydata=None,
            ydata_sigma=None,
            y2data=None,
            args_initial_guess=None,
            args_bounds=None,
            allow_fit=True,
    ):
        self.state_id_tuple = state_id_tuple
        self.transition_function = transition_function
        self.cohort = cohort
        self.args = args
        self.xdata = xdata
        self.ydata = ydata
        self.ydata_sigma = ydata_sigma
        self.y2data = y2data
        self.args_initial_guess = args_initial_guess
        self.args_bounds = args_bounds
        self.allow_fit = allow_fit
        self.original_args = args

    def __repr__(self):
        return 'MarkovTransitionFunction(cohort={}, state_id_tuple={})'.format(self.cohort, self.state_id_tuple)

    def value_at_time_step(self, time_step):
        """return the value of transition function at time_step using args if provided"""
        if self.args is not None:
            return self.transition_function(time_step, *self.args)

        return self.transition_function(time_step)

    def fit_to_data(self, update_args=True):
        """fit transition function to data and optionally update the args based on the output
        Uses curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

        Keyword arguments:
            update_args -- optional, boolean value for whether or not to update args instance variable

        Returns:
            popt -- optimal parameters found for the transition function, `f(xdata, *popt) - ydata` is minimized
        """
        if not self.allow_fit:  # if we don't want to fit this function, return None
            return

        if not self.args_bounds:  # if we didn't provide bounds, use negative and positive infinity
            bounds = (-np.inf, np.inf)
        else:  # otherwise, use the provided bounds
            bounds = self.args_bounds

        absolute_sigma = None
        if self.ydata_sigma is not None:
            absolute_sigma = False

        popt, pcov = curve_fit(
            self.transition_function,
            xdata=self.xdata,
            ydata=self.ydata,
            p0=self.args_initial_guess,
            sigma=self.ydata_sigma,
            absolute_sigma=absolute_sigma,
            bounds=bounds,
        )
        if update_args:
            self.args = popt

        return popt

    def plot_actual_vs_args(self, file_path=None, y2=False):
        x = self.xdata
        y = self.ydata

        if y2:
            y = self.y2data

        y_fit = self.transition_function(x, *self.args)
        if file_path:
            plot([
                go.Scatter(x=x, y=y, name='actual'),
                go.Scatter(x=x, y=y_fit, name='fit'),
            ], filename=file_path)
        else:
            plot([
                go.Scatter(x=x, y=y, name='actual'),
                go.Scatter(x=x, y=y_fit, name='fit'),
            ])
        return


if __name__ == '__main__':
    def mx_plus_b(t, slope=1, intercept=0):
        return t * slope + intercept

    test_x = np.array([0, 1, 2, 3, 4, 5])
    test_y = np.array([0, 2, 4, 6, 8, 10])

    tf = MarkovTransitionFunction(
        state_id_tuple=('hello', 'world'),
        transition_function=mx_plus_b,
        xdata=test_x,
        ydata=test_y,
        args_initial_guess=[4, 10],
        args_bounds=([0, 0], [100, 100]),
    )

    print(tf.value_at_time_step(time_step=4))
    print(tf)
    print(tf.fit_to_data(update_args=True))

    print(tf.value_at_time_step(time_step=4))
