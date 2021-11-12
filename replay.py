from collections import namedtuple
import torch
import numpy as np

# Batch namedtuple, i.e. a class which contains the given attributes
Batch = namedtuple(
    'Batch', ('states', 'actions', 'rewards', 'next_states', 'dones')
)


class ReplayMemory:
    def __init__(self, max_size, state_size):
        """Replay memory implemented as a circular buffer.

        Experiences will be removed in a FIFO manner after reaching maximum
        buffer size.

        Args:
            - max_size: Maximum size of the buffer.
            - state_size: Size of the state-space features for the environment.
        """
        self.max_size = max_size
        self.state_size = state_size

        # Pre-allocating all the required memory, for speed concerns
        self.states = torch.empty((max_size, state_size))
        self.actions = torch.empty((max_size, 1), dtype=torch.long)
        self.rewards = torch.empty((max_size, 1))
        self.next_states = torch.empty((max_size, state_size))
        self.dones = torch.empty((max_size, 1), dtype=torch.bool)

        # pointer to the current location in the circular buffer
        self.idx = 0
        # indicates number of transitions currently stored in the buffer
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer.

        :param state:  1-D np.ndarray of state-features.
        :param action:  integer action.
        :param reward:  float reward.
        :param next_state:  1-D np.ndarray of state-features.
        :param done:  boolean value indicating the end of an episode.
        """
        # Store the input values into the appropriate attributes
        self.states[self.idx] = torch.from_numpy(state)
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = torch.from_numpy(next_state)
        self.dones[self.idx] = done

        # circulate the pointer to the next position
        self.idx = (self.idx + 1) % self.max_size

        # update the current buffer size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size) -> Batch:
        """Sample a batch of experiences.

        If the buffer contains less that `batch_size` transitions, sample all
        of them.

        :param batch_size:  Number of transitions to sample.
        :rtype: Batch
        """
        # Randomly sample an appropriate number of
        # transitions without replacement.  If the buffer contains less than
        # `batch_size` transitions, return all of them.

        if batch_size > self.size:
            sample_indices = np.arange(self.size)
            np.random.shuffle(sample_indices)
        else:
            sample_indices = np.random.choice(range(self.size), size=batch_size, replace=False)

        batch = Batch(self.states[sample_indices], self.actions[sample_indices],
                      self.rewards[sample_indices], self.next_states[sample_indices],
                      self.dones[sample_indices])

        return batch

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env:  Openai Gym environment
        :param num_steps:  Number of steps to populate the
        """
        # Run a random policy for `num_steps` time-steps and
        # populate the replay memory with the resulting transitions.

        state = env.reset()
        for _ in range(num_steps):
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            self.add(state, action, reward, next_state, done)

            if done:
                state = env.reset()
            else:
                state = next_state
