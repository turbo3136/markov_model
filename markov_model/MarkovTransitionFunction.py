import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objs as go
from plotly.offline import plot


class MarkovTransitionFunction:
    """Create a MarkovTransitionFunction object defining the probability of transitioning between two MarkovStates
    Keyword arguments:
        cohort -- identifier for the cohort, usually a datetime object
        state_id_tuple -- unique identifier, tuple containing the id of the initial state and the end state
        transition_function -- time dependent function representing the transition probability between
            initial and end states. transition_function's first argument must be t (time_step),
            followed by optional args
        args -- optional, values of optional arguments in transition function
        xdata -- optional, numpy array of x axis data to be used for curve_fit
        ydata -- optional, numpy array of y axis data to be used for curve_fit
        ydata_sigma -- optional, numpy array describing the variance of ydata
        args_initial_guess -- optional, initial guess for args, used for fitting transition function to data
        args_bounds -- optional, 2-tuple of list, lower an upper bounds used for fitting transition function to data
        allow_fit -- optional, boolean for whether to allow fitting for this transition function
        is_remainder -- optional, boolean for whether this function should be 1 - (other transition probabilities),
            note that the value in a MarkovTransitionMatrix.value_at_time_step will be the remainder
    """

    def __init__(
            self,
            cohort,
            state_id_tuple,
            transition_function,
            args=None,
            xdata=None,
            ydata=None,
            ydata_sigma=None,
            args_initial_guess=None,
            args_bounds=None,
            allow_fit=True,
            is_remainder=False,
    ):
        self.state_id_tuple = state_id_tuple
        self.transition_function = transition_function
        self.cohort = cohort
        self.args = args
        self.xdata = xdata
        self.ydata = ydata
        self.ydata_sigma = ydata_sigma
        self.args_initial_guess = args_initial_guess
        self.args_bounds = args_bounds
        self.allow_fit = allow_fit
        self.original_args = args
        self.is_remainder = is_remainder

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
        if self.allow_fit != 1:  # if we don't want to fit this function, return None
            return

        if not self.args_bounds:  # if we didn't provide bounds, use negative and positive infinity
            bounds = (-np.inf, np.inf)
        else:  # otherwise, use the provided bounds
            bounds = self.args_bounds

        absolute_sigma = None
        if self.ydata_sigma is not None:
            absolute_sigma = True

        # TODO: figure out a better solution here
        try:
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
        except ValueError:
            print('Fitting failed for cohort {}, tuple {}'.format(self.cohort, self.state_id_tuple))
            return

    def plot_actual_vs_args(self, file_path=None, auto_open=False):
        # TODO: check that this data is a numpy array
        x = self.xdata
        y = self.ydata

        y_fit = self.transition_function(x, *self.args)
        if file_path:
            plot([
                go.Scatter(x=x, y=y, name='actual'),
                go.Scatter(x=x, y=y_fit, name='fit'),
            ], filename=file_path, auto_open=auto_open)
        else:
            plot([
                go.Scatter(x=x, y=y, name='actual'),
                go.Scatter(x=x, y=y_fit, name='fit'),
            ], auto_open=auto_open)
        return
