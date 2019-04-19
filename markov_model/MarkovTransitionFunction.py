import numpy as np
from scipy.optimize import curve_fit


class MarkovTransitionFunction:
    """Create a MarkovTransitionFunction object defining the probability of transitioning between two MarkovStates

    Keyword arguments:
        state_id_tuple -- unique identifier, tuple containing the id of the initial state and the end state
        transition_function -- time dependent function representing the transition probability between
            initial and end states. transition_function's first argument must be t (time_step),
            followed by optional args
        args -- optional, values of optional arguments in transition function
        args_initial_guess -- optional, initial guess for args, used for fitting transition function to data
        args_bounds -- optional, 2-tuple of list, lower an upper bounds used for fitting transition function to data
        allow_fit -- optional, boolean for whether to allow fitting for this transition function
    """

    def __init__(
            self,
            state_id_tuple,
            transition_function,
            args=None,
            args_initial_guess=None,
            args_bounds=None,
            allow_fit=True,
    ):
        self.state_id_tuple = state_id_tuple
        self.transition_function = transition_function
        self.args = args
        self.args_initial_guess = args_initial_guess
        self.args_bounds = args_bounds
        self.allow_fit = allow_fit

    def __repr__(self):
        return 'MarkovTransitionFunction(state_id_tuple={})'.format(self.state_id_tuple)

    def value_at_time_step(self, time_step):
        """return the value of transition function at time_step using args if provided"""
        if self.args is not None:
            return self.transition_function(time_step, *self.args)

        return self.transition_function(time_step)

    def fit_to_data(self, xdata, ydata, update_args=True):
        """fit transition function to data and optionally update the args based on the output
        Uses curve_fit: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html

        Keyword arguments:
            xdata -- np array of x values
            ydata -- np array of y values to fit parameters to
            update_args -- optional, boolean value for whether or not to update args class variable

        Returns:
            popt -- optimal parameters found for the transition function, `f(xdata, *popt) - ydata` is minimized
        """
        if not self.allow_fit:  # if we don't want to fit this function, return None
            return

        if not self.args_bounds:  # if we didn't provide bounds, use negative and positive infinity
            bounds = (-np.inf, np.inf)
        else:  # otherwise, use the provided bounds
            bounds = self.args_bounds

        popt, pcov = curve_fit(
            self.transition_function,
            xdata=xdata,
            ydata=ydata,
            p0=self.args_initial_guess,
            bounds=bounds,
        )
        if update_args:
            self.args = popt

        return popt


if __name__ == '__main__':
    def mx_plus_b(t, slope=1, intercept=0):
        return t * slope + intercept

    tf = MarkovTransitionFunction(
        state_id_tuple=('hello', 'world'),
        transition_function=mx_plus_b,
        args_initial_guess=[4, 10],
        args_bounds=([0, 0], [100, 100]),
    )

    print(tf.value_at_time_step(time_step=4))

    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 2, 4, 6, 8, 10])
    print(tf)
    print(tf.fit_to_data(xdata=x, ydata=y, update_args=True))

    print(tf.value_at_time_step(time_step=4))
