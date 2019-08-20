import numpy as np
import helpers


class MarkovStateVector:
    """Create a MarkovStateVector object defining a probability distribution within the state space.

    Keyword arguments:
        state_space -- MarkovStateSpace object for this system, i.e. a list of all possible MarkovState(s)
        state_distribution -- numpy array representing the probability distribution within the state space
        time_step -- time step for the system this vector represents
    """

    def __init__(self, cohort, state_space, state_distribution, time_step, time_step_interval, size=1):
        if type(state_distribution) != np.ndarray:
            raise ValueError('MarkovStateVector.state_distribution expects a numpy ndarray object')

        self.cohort = cohort
        self.state_space = state_space
        self.state_distribution = state_distribution
        self.time_step = time_step
        self.initial_time_step = time_step
        self.time_step_interval = time_step_interval
        self.size = size

        self.current_date = helpers.add_interval_to_date(
            date_object=helpers.date_string_to_datetime(self.cohort),
            steps=self.time_step,
            interval=self.time_step_interval,
        )

        self.total_state_distribution_size = sum(self.state_distribution)

        self.state_distribution_dict = {
            state_id: self.state_distribution[index] for index, state_id in enumerate(self.state_space.state_id_list)
        }

        if self.state_space.size != len(self.state_distribution):
            raise ValueError(
                'MarkovStateVector.state_distribution must be the same size as state_space array'
            )

    def __repr__(self):
        return 'MarkovStateVector(cohort={}, state_space={}, state_distribution={}, time_step={})'.format(
            self.cohort, self.state_space, self.state_distribution, self.time_step
        )
