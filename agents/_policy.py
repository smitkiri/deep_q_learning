import numpy as np
from collections import defaultdict
from scheduler import ExponentialSchedule
from typing import Callable, Sequence, List


def _argmax(arr: Sequence[float]) -> List[int]:
    """Argmax that breaks ties randomly

    Takes in a list of values and returns the index of the item with the highest value, breaking ties randomly.

    Note: np.argmax returns the first index that matches the maximum,
    so we define this method to use in EpsilonGreedy policy.
    Args:
        arr: sequence of values
    """
    arr = np.array(arr)

    # Get all the indices of the maximum value
    max_indices = np.where(arr == arr.max())[0]

    return max_indices


def create_epsilon_policy(Q: defaultdict, epsilon_scheduler: ExponentialSchedule,
                          return_probs: bool = False) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple.
    More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon_scheduler (ExponentialSchedule): softness scheduler
        return_probs (bool): Indicates if the policy should return probabilities
                             of actions or just the action.
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Sequence, step: int) -> int:
        if not isinstance(state, tuple):
            state = tuple(state)

        epsilon = epsilon_scheduler.value(step)

        # Make sure to break ties arbitrarily
        if np.random.random() < epsilon:
            action = np.random.choice(range(0, num_actions))
        else:
            action_list = _argmax(Q[state])
            action = np.random.choice(action_list)

        return action

    def get_action_probs(state: Sequence, step: int) -> np.ndarray:
        if not isinstance(state, tuple):
            state = tuple(state)

        epsilon = epsilon_scheduler.value(step)

        # Action probabilities of non-greedy actions
        action_probs = np.ones(num_actions) * (epsilon / num_actions)

        # Get the greedy action
        greedy_action = np.argmax(Q[state])

        # Update the probability of the greedy action
        action_probs[greedy_action] += 1 - epsilon

        return action_probs

    if return_probs:
        return get_action_probs

    return get_action
