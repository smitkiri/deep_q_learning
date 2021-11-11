from collections import defaultdict
import numpy as np
import math
import gym

from agents._policy import create_epsilon_policy
from scheduler import ExponentialSchedule
from tqdm import trange
import datetime
import pickle
import os

from typing import List, Sequence, Union, Tuple


class CartPoleQAgent:
    """Agent for applying the Q-Learning algorithm on the CartPole environment"""

    def __init__(
            self, num_steps: int,
            gamma: float,
            max_epsilon: float,
            min_epsilon: float,
            min_step_size: float,
            max_step_size: float,
            buckets: Sequence[int]
    ):
        """
        :param num_steps: The total number of steps for the agent to train
        :param gamma: Discount factor
        :param max_epsilon: The epsilon value of e-greedy policy
        :param min_step_size: The step size to update Q-values
        :param buckets: The number of buckets for each value in the state - to convert the state to discrete
        """
        # Define the hyperparameters
        self.num_steps = num_steps
        self.gamma = gamma
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size

        # Create schedulers
        self.eps_scheduler = ExponentialSchedule(value_from=max_epsilon, value_to=min_epsilon, num_steps=num_steps)
        self.alpha_scheduler = ExponentialSchedule(value_from=max_step_size, value_to=min_step_size, num_steps=num_steps)

        # Define the environment and initialize the Q values + epsilon greedy policy
        self.env = gym.make("CartPole-v1")
        self.Q = defaultdict(lambda: np.zeros(self.env.action_space.n))
        self.policy = create_epsilon_policy(self.Q, epsilon_scheduler=self.eps_scheduler)

        # Define the upper and lower bounds of the discrete observation space
        self.upper_bounds = (self.env.observation_space.high[0], 0.5,
                             self.env.observation_space.high[2], math.radians(50))

        self.lower_bounds = (self.env.observation_space.low[0], -0.5,
                             self.env.observation_space.low[2], -math.radians(50))

        # Set the number of buckets in the discrete observation space
        self.num_buckets = buckets

        # Flag to indicate if the agent is trained
        self.is_trained = False

        # Training history
        self.episode_returns = None
        self.episode_lengths = None

    def get_discrete_state(self, state: List[float]) -> Tuple[int]:
        """
        Converts the continuous state to a discrete one

        :param state: The continuous state
        :return: The discrete state
        """
        discrete_state = []

        for idx in range(len(state)):
            # Get the discrete value from the continuous by putting them into a specific bucket
            ratio = (abs(self.lower_bounds[idx]) + state[idx]) / (self.upper_bounds[idx] - self.lower_bounds[idx])
            discrete_val = round(ratio * (self.num_buckets[idx] - 1))

            # Check if the bucket is out of bounds, if so put them in the first / last bucket
            discrete_val = min(self.num_buckets[idx] - 1, max(0, discrete_val))
            discrete_state.append(discrete_val)

        return tuple(discrete_state)

    def save_trained_results(self, output_dir: Union[str, os.PathLike] = None) -> str:
        """
        Saves the trained results

        :param output_dir:
        :return: Output file name
        """
        if output_dir is None:
            output_dir = "model_outputs/"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        curr_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        output_filename = curr_datetime + "_qlearning_training_history.pickle"
        output_path = os.path.join(output_dir, output_filename)

        data_dump = {
            "q_values": dict(self.Q),
            "episode_lengths": self.episode_lengths,
            "episode_returns": self.episode_returns
        }

        with open(output_path, "wb") as f:
            pickle.dump(data_dump, f)

        return output_path

    def run(self, output_dir: Union[str, os.PathLike] = None) -> None:
        """
        Run the agent
        :return:
        """
        if self.is_trained:
            print("The agent is already trained.")
            return

        # Initialize to store returns and episode lengths
        rewards = []
        returns = []
        lengths = []

        episode_num = 0
        episode_time_step = 0

        # Initialize the state and action based on the policy
        state = self.env.reset()
        state = self.get_discrete_state(state)

        for step in trange(self.num_steps, desc="Step", leave=False):
            # Take an action based on the current policy
            action = self.policy(state, step)

            # Take a step in the environment and observe S' and R
            new_state, reward, done, _ = self.env.step(action)
            new_state = self.get_discrete_state(new_state)
            rewards.append(reward)

            step_size = self.alpha_scheduler.value(step)

            # Update Q-values
            self.Q[state][action] += step_size * (
                    reward + self.gamma * max(self.Q[new_state]) - self.Q[state][action])

            if done:
                # Compute the return for the episode that just ended
                episode_return = 0
                for r in rewards[::-1]:
                    episode_return = r + self.gamma * episode_return

                returns.append(episode_return)
                lengths.append(episode_time_step)

                # Reset the variables to start a new episode
                state = self.env.reset()
                state = self.get_discrete_state(state)
                episode_time_step = 0
                episode_num += 1
                rewards = []
            else:
                state = new_state
                episode_time_step += 1

        print("\nTraining finished.")

        self.is_trained = True
        self.episode_returns = np.array(returns)
        self.episode_lengths = np.array(lengths)

        dump_loc = self.save_trained_results(output_dir)
        print(f"Training results saved at {dump_loc}")

        return



