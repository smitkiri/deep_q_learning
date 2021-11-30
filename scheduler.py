import numpy as np


class ExponentialSchedule:
    def __init__(self, value_from: float, value_to: float, eps_decay: float = None, num_steps: int = None):
        """
        Decay from `value_from` to `value_to` with `eps_decay ^ step_number`

        :param value_from: initial value
        :param value_to: final value
        :param eps_decay: decay rate
        :param num_steps: The number of steps to get to `value_to`
        """
        self.value_from = value_from
        self.value_to = value_to
        self.num_steps = num_steps

        if self.num_steps is not None:
            self.eps_decay = np.exp(np.log(self.value_to) / self.num_steps)
        else:
            self.eps_decay = eps_decay

    def value(self, episode_num: int) -> float:
        """
        Get the epsilon value for the current episode

        :param episode_num: Current episode number
        :return: Epsilon
        """
        value = self.value_from * (self.eps_decay ** episode_num)

        if value < self.value_to:
            value = self.value_to

        return value
