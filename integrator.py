import numpy as np

class IntegratorSuite:
    """
    From-scratch integrator
    """
    def __init__(self, equation, initial_value, lower_bound, upper_bound):
        self.equation = equation
        self.initial = initial_value
        self.lower = lower_bound
        self.upper = upper_bound

    def euler(self, equation, initial_value, lower_bound, upper_bound, step_size=1e-6):
        """
        Integrate using Euler method
        """
        current_x = lower_bound
        current_y = initial_value
        steps = (upper_bound - lower_bound) / step_size
        for _ in np.arange(steps):
            current_y += step_size * equation(current_x, current_y)
            current_x += step_size
        return current_y

    def rk2(self, step_size=1e-6):
        """
        Integrate using Runge-Kutta second order method
        """
        current_x = self.lower
        current_y = self.initial
        steps = (self.upper - self.lower) / step_size
        for _ in np.arange(steps):
            k1 = self.equation(current_x, current_y)
            k2 = self.equation(current_x + step_size * 0.5, current_y + step_size * k1 * 0.5)
            current_y += step_size * k2
            current_x += step_size
        return current_y

    def rk4(self, step_size=1e-6):
        """
        Integrate using Runge-Kutta fourth-order method
        """
        current_x = self.lower
        current_y = self.initial
        steps = (self.upper - self.lower) / step_size
        for _ in np.arange(steps):
            k1 = self.equation(current_x, current_y)
            k2 = self.equation(current_x + step_size * 0.5, current_y + step_size * k1 * 0.5)
            k3 = self.equation(current_x + step_size * 0.5, current_y + step_size * k2 * 0.5)
            k4 = self.equation(current_x + step_size, current_y + step_size * k3)
            current_y += (step_size / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
            current_x += step_size
        return current_y
