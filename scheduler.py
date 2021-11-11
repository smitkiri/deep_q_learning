import numpy as np


class ExponentialSchedule:
    def __init__(self, value_from: float, value_to: float, num_steps: int):
        """Exponential schedule from `value_from` to `value_to` in `num_steps` steps.

        $value(t) = a \exp (b t)$

        :param value_from: initial value
        :param value_to: final value
        :param num_steps: number of steps for the exponential schedule
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        self.a = value_from
        self.b = (np.log(value_to) - np.log(self.a)) / (num_steps - 1)

    def value(self, step: int) -> float:
        """Return exponentially interpolated value between `value_from` and `value_to`interpolated value between.

        returns {
            `value_from`, if step == 0 or less
            `value_to`, if step == num_steps - 1 or more
            the exponential interpolation between `value_from` and `value_to`, if 0 <= steps < num_steps
        }

        :param step:  The step at which to compute the interpolation.
        :rtype: float.  The interpolated value.
        """

        if step <= 0:
            value = self.value_from
        elif step >= self.num_steps:
            value = self.value_to
        else:
            value = self.a * np.exp(self.b * step)

        return value
