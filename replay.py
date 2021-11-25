from collections import namedtuple, deque
import random

# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple(
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones')
)

Transition = namedtuple(
    'Transition', ('state', 'action', 'reward', 'next_state', 'done')
)


class ReplayMemory:
    def __init__(self, max_size):
        """
        Replay Memory initialized as a circular buffer

        :param max_size: The size of the Memory
        """
        self.memory = deque([], maxlen=max_size)
        self.max_size = max_size

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.

        :param state:  1-D np.ndarray of state-features.
        :param action:  integer action.
        :param reward:  float reward.
        :param next_state:  1-D np.ndarray of state-features.
        :param done:  boolean value indicating the end of an episode.
        """
        transition = Transition(state, action, reward, next_state, done)
        self.memory.append(transition)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences.

            If the buffer contains less that `batch_size` transitions, sample all
            of them.

            :param batch_size:  Number of transitions to sample.
            :rtype: Batch
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        transitions = random.sample(self.memory, k=batch_size)
        batch = Batch(*zip(*transitions))
        return batch
