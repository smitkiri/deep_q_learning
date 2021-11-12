from scheduler import ExponentialSchedule
from replay import ReplayMemory
from networks import DQN

from typing import Tuple, Union
import datetime
import json
import tqdm
import os

import torch.nn.functional as F
import numpy as np
import torch
import gym


class CartPoleDqnAgent:
    """Agent for applying the Q-Learning algorithm on the CartPole environment"""

    def __init__(
            self, num_steps: int,
            gamma: float,
            epsilon_range: Tuple[int, int],
            replay_size: int,
            replay_prepopulate_steps: int = 0,
            batch_size: int = 64
    ):
        """
        :param num_steps: The total number of steps for the agent to train
        :param gamma: Discount factor
        :param epsilon_range: The epsilon value to begin and end with for the e-greedy policy
        :param replay_size: Maximum size of the replay memory
        :param replay_prepopulate_steps: Number of steps with which to pre-populate the replay memory
        :param batch_size: Number of experiences in a batch
        """
        # Define the hyperparameters
        self.num_steps = num_steps
        self.gamma = gamma
        self.epsilon_range = epsilon_range
        self.batch_size = batch_size

        # Create schedulers
        self.eps_scheduler = ExponentialSchedule(
            value_from=max(epsilon_range), value_to=min(epsilon_range), num_steps=num_steps
        )

        # Define the environment and initialize the Q values + epsilon greedy policy
        self.env = gym.make("CartPole-v1")

        # get the state_size from the environment
        state_size = self.env.observation_space.shape[0]

        # initialize the DQN and DQN-target models
        self.dqn_model = DQN(state_size, self.env.action_space.n)
        self.dqn_target = DQN.custom_load(self.dqn_model.custom_dump())

        # initialize the optimizer
        self.optimizer = torch.optim.Adam(self.dqn_model.parameters())

        # initialize the replay memory and pre-populate it
        self.memory = ReplayMemory(replay_size, state_size)
        self.memory.populate(self.env, replay_prepopulate_steps)
        self.replay_prepopulate_steps = replay_prepopulate_steps

        # Flag to indicate if the agent is trained
        self.is_trained = False

        # Training history
        self.episode_returns = None
        self.episode_lengths = None
        self.step_losses = None

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
        model_dir = os.path.join(output_dir, curr_datetime + "_dqn_model")
        os.mkdir(model_dir)

        model_path = os.path.join(model_dir, "model.pt")
        params_path = os.path.join(model_dir, "params.json")

        model_dump = self.dqn_model.custom_dump()
        torch.save(model_dump, model_path)

        params = {
            "num_steps": self.num_steps,
            "gamma": self.gamma,
            "epsilon_range": self.epsilon_range,
            "replay_size": self.memory.max_size,
            "replay_prepopulate_steps": self.replay_prepopulate_steps,
            "batch_size": self.batch_size
        }

        with open(params_path, "w") as f:
            json.dump(params, f)

        return model_dir

    @staticmethod
    def train_dqn_batch(optimizer, batch, dqn_model, dqn_target, gamma) -> float:
        """Perform a single batch-update step on the given DQN model.

        :param optimizer: nn.optim.Optimizer instance.
        :param batch:  Batch of experiences (class defined earlier).
        :param dqn_model:  The DQN model to be trained.
        :param dqn_target:  The target DQN model, ~NOT~ to be trained.
        :param gamma:  The discount factor.
        :rtype: float  The scalar loss associated with this batch.
        """
        # compute the values and target_values tensors using the
        # given models and the batch of data.
        values = dqn_model(batch.states)

        # Get the values for the specific action in batch.actions
        values = values.gather(1, batch.actions)

        # Get the list of all non-final next states
        non_final_next_states = torch.stack([batch.next_states[idx] for idx in range(len(batch.next_states))
                                             if not batch.dones[idx]])

        # Get the Q-values for each action from the target network
        target_preds = dqn_target(non_final_next_states)

        # Set the target values of final states to 0 and fill in all the other non-final values
        target_values = torch.zeros(batch.rewards.shape)
        target_values[~batch.dones] = torch.max(target_preds, dim=1)[0].detach()
        target_values = batch.rewards + gamma * target_values

        assert (
                values.shape == target_values.shape
        ), 'Shapes of values tensor and target_values tensor do not match.'

        # testing that the value tensor requires a gradient,
        # and the target_values tensor does not
        assert values.requires_grad, 'values tensor should not require gradients'
        assert (
            not target_values.requires_grad
        ), 'target_values tensor should require gradients'

        # computing the scalar MSE loss between computed values and the TD-target
        loss = F.mse_loss(values, target_values)

        optimizer.zero_grad()  # reset all previous gradients
        loss.backward()  # compute new gradients
        optimizer.step()  # perform one gradient descent step

        return loss.item()

    def run(self, output_dir: Union[str, os.PathLike] = None) -> None:
        """
        Run the DQN Agent
        """
        # initiate lists to store returns, lengths and losses
        rewards = []
        returns = []
        lengths = []
        losses = []

        episode_num = 0
        episode_time_step = 0
        loss = 0

        state = self.env.reset()  # initialize state of first episode

        # iterate for a total of `num_steps` steps
        pbar = tqdm.trange(self.num_steps)
        for t_total in pbar:
            #  * sample an action from the DQN using epsilon-greedy
            #  * use the action to advance the environment by one step
            #  * store the transition into the replay memory
            eps = self.eps_scheduler.value(t_total)

            if np.random.random() < eps:
                action = self.env.action_space.sample()
            else:
                action_tensor = self.dqn_model(torch.from_numpy(state).unsqueeze(0))
                _, action = torch.max(action_tensor, dim=1)
                action = action.item()

            next_state, reward, done, _ = self.env.step(action)
            self.memory.add(state, action, reward, next_state, done)
            state = next_state

            rewards.append(reward)

            # Once every 4 steps, sample a batch from the replay memory and perform a batch update
            if t_total % 4 == 0:
                batch = self.memory.sample(self.batch_size)
                loss = CartPoleDqnAgent.train_dqn_batch(self.optimizer, batch, self.dqn_model,
                                                        self.dqn_target, self.gamma)

            losses.append(loss)

            # Once every 10_000 steps, update the target network
            if t_total % 10_000 == 0:
                state_dict = self.dqn_model.state_dict()
                self.dqn_target.load_state_dict(state_dict)

            if done:
                # Calculate the episode return
                episode_return = 0
                for r in rewards[::-1]:
                    episode_return = r + self.gamma * episode_return

                returns.append(episode_return)
                lengths.append(episode_time_step)

                # Update the progress bar
                pbar.set_description(
                    f'Episode: {episode_num} | Steps: {episode_time_step + 1} | Return:'
                    f' {episode_return:5.2f} | Epsilon: {eps:4.2f}'
                )

                # Reset variables
                episode_num += 1
                episode_time_step = 0
                rewards = []
                state = self.env.reset()

            else:
                episode_time_step += 1

        self.episode_lengths = np.array(lengths)
        self.episode_returns = np.array(returns)
        self.step_losses = np.array(losses)

        self.is_trained = True

        # Save the model
        dump_loc = self.save_trained_results(output_dir)
        print(f"Training results saved at {dump_loc}")

        return
