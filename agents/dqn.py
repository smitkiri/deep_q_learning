from tqdm import trange

from scheduler import ExponentialSchedule
from replay import ReplayMemory
from networks import DQN, DQNConvNet
from plotting import plot_lengths_returns

from typing import Tuple, Union, List
import datetime
import copy
import pickle
import json
import tqdm
import os

from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import torch
import gym


class DQNAgent:
    """Abstract class for DQN Agent Variants"""

    def __init__(
            self, num_episodes: int,
            gamma: float,
            epsilon_range: Tuple[int, int],
            eps_decay: float,
            eps_num_steps: int,
            batch_size: int,
            target_update: int,
            policy_update: int,
            learning_rate: float,
            step_lr_params: Tuple[int, float],
    ):
        """

        :param num_episodes: The total number of steps for the agent to train
        :param gamma: Discount factor
        :param epsilon_range: The epsilon value to begin and end with for the e-greedy policy
        :param batch_size: Number of experiences in a batch
        """
        # Define the hyperparameters
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.epsilon_range = epsilon_range
        self.batch_size = batch_size
        self.target_update = target_update
        self.policy_update = policy_update
        self.eps_decay = eps_decay
        self.eps_num_steps = eps_num_steps

        # Create schedulers
        self.eps_scheduler = ExponentialSchedule(
            value_from=max(epsilon_range), value_to=min(epsilon_range), eps_decay=eps_decay, num_steps=eps_num_steps
        )

        # Define the environment and set convergence length
        self.env = gym.make("CartPole-v1")
        self.env.seed(0)
        self.converge_len = 450

        # Models and Replay Memory
        self.dqn_model = None
        self.dqn_target = None
        self.optimizer = None
        self.loss_fn = None

        self.memory = None
        self.replay_prepopulate_steps = None
        self.learning_rate = learning_rate
        self.step_lr_params = step_lr_params

        # Image transformation
        self.center_image = None
        self.resize_transform = None

        # Select pyTorch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Flag to indicate if the agent is trained
        self.is_trained = False

        # Training history
        self.episode_returns = None
        self.episode_lengths = None
        self.step_losses = None

    def save_trained_results(self, output_dir: Union[str, os.PathLike] = None, model_name: str = "dqn_model",
                             best_model: dict = None) -> str:
        """
        Saves the trained results

        :param output_dir:
        :param model_name:
        :param best_model:
        :return: Output file name
        """
        if output_dir is None:
            output_dir = "model_outputs/"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        curr_datetime = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        model_dir = os.path.join(output_dir, curr_datetime + "_" + model_name)
        os.mkdir(model_dir)

        model_path = os.path.join(model_dir, "model.pt")
        params_path = os.path.join(model_dir, "params.json")
        training_history_path = os.path.join(model_dir, "training_history.pickle")

        model_dump = self.dqn_model.custom_dump()
        torch.save(model_dump, model_path)

        params = {
            "num_episodes": self.num_episodes,
            "gamma": self.gamma,
            "epsilon_range": self.epsilon_range,
            "eps_decay": self.eps_decay,
            "eps_num_steps": self.eps_num_steps,
            "replay_size": self.memory.max_size,
            "replay_prepopulate_steps": self.replay_prepopulate_steps,
            "batch_size": self.batch_size,
            "target_update": self.target_update,
            "policy_update": self.policy_update,
            "learning_rate": self.learning_rate,
            "step_lr_params": self.step_lr_params,
            "center_image": self.center_image,
        }

        if best_model is not None:
            best_model_path = os.path.join(model_dir, "best_model.pt")
            best_model_dump = best_model["model"].custom_dump()
            torch.save(best_model_dump, best_model_path)

            params["best_model_episodes"] = best_model["episode_num"]

        with open(params_path, "w") as f:
            json.dump(params, f)

        training_history = {
            "loss": self.step_losses,
            "episode_lengths": self.episode_lengths,
            "episode_returns": self.episode_returns
        }

        with open(training_history_path, "wb") as f:
            pickle.dump(training_history, f)

        plot_lengths_returns(self.episode_returns, self.episode_lengths, smooth_line=True,
                             output_file=os.path.join(model_dir, "plot.png"))

        return model_dir


class CartPoleDqnAgent(DQNAgent):
    """Agent for applying the Q-Learning algorithm on the CartPole environment"""

    def __init__(self, input_type: str, **kwargs):
        """
        Initialize the CartPole DQN agent

        :param input_type: The type of input for the neural network. "image" or "vector".
        :param **kwargs: Arguments to initialize the vector agent or the image agent.
        """
        try:
            super(CartPoleDqnAgent, self).__init__(
                num_episodes=kwargs.pop("num_episodes"),
                gamma=kwargs.pop("gamma"),
                epsilon_range=kwargs.pop("epsilon_range"),
                eps_decay=kwargs.pop("eps_decay", None),
                eps_num_steps=kwargs.pop("eps_num_steps", None),
                batch_size=kwargs.pop("batch_size"),
                target_update=kwargs.pop("target_update"),
                policy_update=kwargs.pop("policy_update"),
                learning_rate=kwargs.pop("learning_rate"),
                step_lr_params=kwargs.pop("step_lr_params")
            )

            replay_size = kwargs.pop("replay_size")
            replay_prepopulate_steps = kwargs.pop("replay_prepopulate_steps")
            self.input_type = input_type.lower()

            if input_type.lower() == "vector":
                self.init_vector_agent(
                    replay_size=replay_size,
                    replay_prepopulate_steps=replay_prepopulate_steps,
                    **kwargs
                )

            elif input_type.lower() == "image":
                self.init_image_agent(
                    replay_size=replay_size,
                    replay_prepopulate_steps=replay_prepopulate_steps,
                    center_image=kwargs.pop("center_image", True),
                    **kwargs
                )

            else:
                msg = "The input type could be 'image' or 'vector'."
                raise ValueError(msg)

        except KeyError as missing_arg:
            msg = f"{missing_arg} parameter missing."
            raise KeyError(msg)

    @classmethod
    def load_model(cls, model_dir: Union[str, os.PathLike]):
        """Loads a pre-trained model"""
        params_file = os.path.join(model_dir, "params.json")
        model_file = os.path.join(model_dir, "model.pt")

        with open(params_file, "r") as f:
            params = json.load(f)

        params["replay_prepopulate_steps"] = 0
        input_type = model_dir.split("_")[-1]

        if "input_type" in params:
            input_type = params.pop("input_type")

        center_image = params.pop("center_image")
        agent = cls(input_type, **params)

        with open(model_file, "rb") as f:
            model_data = torch.load(f, map_location=agent.device)

        if input_type == "vector":
            agent.dqn_model = DQN.custom_load(model_data)
        else:
            agent.dqn_model = DQNConvNet.custom_load(model_data)

        agent.dqn_model.to(agent.device)
        agent.dqn_target = None
        agent.center_image = center_image

        return agent

    def init_vector_agent(
            self, replay_size: int,
            replay_prepopulate_steps: int = 0,
            **kwargs
    ):
        """
        Initializes the agent to work with vector data

        :param replay_size: Maximum size of the replay memory
        :param replay_prepopulate_steps: Number of steps with which to pre-populate the replay memory
        :param kwargs: Additional parameters for initializing the Neural Network
        """
        # get the state_size from the environment
        state_size = self.env.observation_space.shape[0]

        # initialize the DQN and DQN-target models
        self.dqn_model = DQN(state_size, self.env.action_space.n, **kwargs).to(self.device)
        self.dqn_target = DQN.custom_load(self.dqn_model.custom_dump()).to(self.device)

        # initialize the optimizer
        self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=self.learning_rate)

        if self.step_lr_params is not None:
            step_size, gamma = self.step_lr_params
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)

        else:
            self.lr_scheduler = None

        self.loss_fn = "mse"

        # initialize the replay memory and pre-populate it
        self.memory = ReplayMemory(replay_size)
        self.populate_memory(replay_prepopulate_steps)
        self.replay_prepopulate_steps = replay_prepopulate_steps

    def init_image_agent(
            self,
            replay_size: int,
            replay_prepopulate_steps: int = 0,
            center_image: bool = True,
            **kwargs
    ):
        """
        Initializes the agent to work with image data

        :param replay_size: Maximum size of the replay memory
        :param replay_prepopulate_steps: Number of steps with which to pre-populate the replay memory
        :param center_image: Whether to crop and center the rendered cartpole image
        :param kwargs: Additional parameters for initializing the Neural Network
        """
        self.env.reset()
        self.center_image = center_image

        # Transformation for image input
        self.resize_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(size=40, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor()
        ])

        init_screen = self.get_screen()
        _, screen_channels, screen_height, screen_width = init_screen.shape

        # Initialize the policy and target DQN models
        self.dqn_model = DQNConvNet(height=screen_height, width=screen_width, channels=screen_channels,
                                    action_dim=self.env.action_space.n, **kwargs).to(self.device)
        self.dqn_target = DQNConvNet.custom_load(self.dqn_model.custom_dump()).to(self.device)
        self.dqn_target.eval()

        # Initialize the optimizer
        self.optimizer = torch.optim.Adam(self.dqn_model.parameters(), lr=self.learning_rate)

        if self.step_lr_params is not None:
            step_size, gamma = self.step_lr_params
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size, gamma)

        else:
            self.lr_scheduler = None

        self.loss_fn = "smooth_l1_loss"

        # Initialize the replay memory and pre-populate it
        self.memory = ReplayMemory(replay_size)
        self.populate_memory(replay_prepopulate_steps)

    def get_cart_location(self, screen_width: int) -> int:
        """
        Gets the cart location based on the screen width (the dimension of the rendered image)

        :param screen_width: The width of the screen
        :return: The cart location relative to the screen width
        """
        # The world is such that 0 is at the center
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        cart_location = int(scale * self.env.state[0] + (screen_width / 2))
        return cart_location

    def get_screen(self) -> torch.Tensor:
        """
        Gets the screen pixels in a torch.Tensor format with a batch dimension added

        :return: (1, Channel, Height, Width) shaped tensor
        """
        # Get the rendered screen and transform it to (C, H, W) order
        screen = self.env.render(mode="rgb_array").transpose((2, 0, 1))

        # Remove the top and bottom of the screen
        _, height, width = screen.shape
        screen = screen[:, int(height * 0.4):int(height * 0.8)]

        if self.center_image:
            # Remove 40% of the edges so we have a centered image of the cart
            view_width = int(width * 0.6)
            cart_location = self.get_cart_location(width)

            if cart_location < view_width // 2:
                slice_range = slice(view_width)
            elif cart_location > width - view_width // 2:
                slice_range = slice(-view_width, None)
            else:
                slice_range = slice(cart_location - view_width // 2, cart_location + view_width // 2)

            screen = screen[:, :, slice_range]

        # Scale the matrix and convert to torch tensor
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        # Resize the image and add a batch dimension
        screen = self.resize_transform(screen).unsqueeze(0)
        screen = screen.to(self.device)

        return screen

    def reset_env(self):
        """
        Resets the current environment

        :return: The state tensor
        """
        if self.input_type == "vector":
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
            last_screen = None
            curr_screen = None
        else:
            self.env.reset()
            last_screen = self.get_screen()
            curr_screen = self.get_screen()
            state = curr_screen - last_screen

        return state, last_screen, curr_screen

    def take_env_step(self, action, last_screen=None, curr_screen=None):
        """
        Take a step in the environment

        :return: next_state, reward, done, last_screen, curr_screen
        """
        if self.input_type == "vector":
            next_state, reward, done, _ = self.env.step(action)
            next_state = torch.tensor(next_state, dtype=torch.float, device=self.device).unsqueeze(0)

        else:
            _, reward, done, _ = self.env.step(action)

            last_screen = curr_screen
            curr_screen = self.get_screen()

            if not done:
                next_state = curr_screen - last_screen
            else:
                next_state = None

        reward = torch.tensor([reward], device=self.device)
        done = torch.tensor([done], device=self.device)

        return next_state, reward, done, last_screen, curr_screen

    def populate_memory(self, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param num_steps:  Number of steps to populate the memory with
        """
        # Run a random policy for `num_steps` time-steps and
        # populate the replay memory with the resulting transitions.
        state, last_screen, curr_screen = self.reset_env()

        for _ in range(num_steps):
            action = self.env.action_space.sample()
            next_state, reward, done, last_screen, curr_screen = self.take_env_step(action, last_screen, curr_screen)
            action = torch.tensor([action], device=self.device)

            self.memory.add(state, action, reward, next_state, done)

            # Reset the states if the current episode is done
            if done:
                state, last_screen, curr_screen = self.reset_env()
            else:
                state = next_state

    def train_dqn_batch(self, batch) -> float:
        """Perform a single batch-update step on the given DQN model.

        :param batch:  Batch of experiences (class defined earlier).
        :rtype: float  The scalar loss associated with this batch.
        """
        # compute the values and target_values tensors using the
        # given models and the batch of data.

        def convert_to_tensor(value: Union[Tuple[torch.Tensor], torch.Tensor], unsqueeze_dim: int = None) -> torch.Tensor:
            """Converts a batch value to tensor if not already"""
            if not isinstance(value, torch.Tensor):
                value = torch.cat(value)

            if unsqueeze_dim is not None:
                value = value.unsqueeze(unsqueeze_dim)

            return value

        states_batch = convert_to_tensor(batch.states)
        actions_batch = convert_to_tensor(batch.actions, unsqueeze_dim=1)
        rewards_batch = convert_to_tensor(batch.rewards, unsqueeze_dim=1)
        dones_batch = convert_to_tensor(batch.dones, unsqueeze_dim=1)
        next_states_batch = batch.next_states

        # Get the policy network predictions
        values = self.dqn_model(states_batch)
        values = values.to(self.device)

        # Get the values for the specific action in batch.actions
        values = values.gather(1, actions_batch)

        # Get the list of all non-final next states
        non_final_next_states = torch.cat([next_states_batch[idx] for idx in range(len(next_states_batch))
                                           if not batch.dones[idx]])

        # Get the Q-values for each action from the target network
        target_preds = self.dqn_target(non_final_next_states)

        # Set the target values of final states to 0 and fill in all the other non-final values
        target_values = torch.zeros(rewards_batch.shape, device=self.device)
        target_values[~dones_batch] = torch.max(target_preds, dim=1)[0].detach()
        target_values = rewards_batch + self.gamma * target_values

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
        if self.loss_fn == "mse":
            loss = F.mse_loss(values, target_values)
        elif self.loss_fn == "smooth_l1_loss":
            loss = F.smooth_l1_loss(values, target_values)
        else:
            msg = f"loss_fn argument should either be 'mse' or 'smooth_l1_loss'."
            raise ValueError(msg)

        # reset all previous gradients
        self.optimizer.zero_grad()

        # compute new gradients
        loss.backward()

        # Perform gradient clipping
        for param in self.dqn_model.parameters():
            param.grad.data.clamp_(-1, 1)

        # perform one gradient descent step
        self.optimizer.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        return loss.item()

    def select_action(self, state: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Selects an e-greedy action to take.
        """
        if np.random.random() < eps:
            action = self.env.action_space.sample()
            action = torch.tensor([action], device=self.device)
        else:
            with torch.no_grad():
                action_tensor = self.dqn_model(state)
                _, action = torch.max(action_tensor, dim=1)

        return action

    def test_model(self, n_episodes: int = 10_000, seed: int = 0) -> List[int]:
        """Test the DQN model on random episode initializations and return the number
        of time steps in each episode
        """
        self.env = gym.make("CartPole-v1")
        self.env.seed(seed)
        self.dqn_model.eval()

        time_step_hist = []

        for _ in trange(n_episodes, desc="Episode"):
            curr_time_step = 0
            state, last_screen, curr_screen = self.reset_env()

            while True:
                action = self.select_action(state, eps=0)
                next_state, reward, done, last_screen, curr_screen = self.take_env_step(
                    action.item(), last_screen, curr_screen
                )

                state = next_state
                curr_time_step += 1

                if done:
                    time_step_hist.append(curr_time_step)
                    break

        return time_step_hist

    def run(self, output_dir: Union[str, os.PathLike] = None) -> None:
        """
        Run the DQN Agent
        """
        # initiate lists to store returns, lengths and losses
        returns = []
        lengths = []
        losses = []

        t_total = 0
        loss = 0
        img = None

        best_model = {"avg_len": 0, "episode_num": 0, "model": None}

        # iterate for a total of `num_steps` steps
        pbar = tqdm.trange(self.num_episodes)
        for episode_num in pbar:
            # initialize state of first episode
            state, last_screen, curr_screen = self.reset_env()

            episode_time_step = 0
            avg_len = 0
            rewards = []
            eps = self.eps_scheduler.value(episode_num)

            while True:
                #  * sample an action from the DQN using epsilon-greedy
                #  * use the action to advance the environment by one step
                #  * store the transition into the replay memory
                action = self.select_action(state, eps)

                next_state, reward, done, last_screen, curr_screen = self.take_env_step(
                    action.item(), last_screen, curr_screen
                )

                self.memory.add(state, action, reward, next_state, done)
                state = next_state
                t_total += 1

                # Sample a batch from the replay memory and perform a batch update
                if t_total % self.policy_update == 0:
                    batch = self.memory.sample(self.batch_size)
                    loss = self.train_dqn_batch(batch)

                losses.append(loss)
                rewards.append(reward.item())

                # Update the target network
                if t_total % self.target_update == 0:
                    state_dict = self.dqn_model.state_dict()
                    self.dqn_target.load_state_dict(state_dict)
                    self.dqn_target.to(self.device)

                if done:
                    # Calculate the episode return
                    episode_return = 0
                    for r in rewards[::-1]:
                        episode_return = r + self.gamma * episode_return

                    returns.append(episode_return)
                    lengths.append(episode_time_step)

                    if episode_num > 100:
                        avg_len = round(np.mean(lengths[-100:]))

                    # Update the progress bar
                    pbar.set_description(
                        f'Episode: {episode_num} | Steps: {episode_time_step + 1} | Return:'
                        f' {episode_return:5.2f} | Epsilon: {eps:4.2f} | Avg len: {avg_len}'
                    )
                    break

                else:
                    episode_time_step += 1

            if avg_len > best_model["avg_len"]:
                best_model["model"] = copy.deepcopy(self.dqn_model)
                best_model["episode_num"] = episode_num

            if avg_len >= self.converge_len:
                print(f"The model converged in {episode_num} episodes.")
                self.num_episodes = episode_num
                break

        self.episode_lengths = np.array(lengths)
        self.episode_returns = np.array(returns)
        self.step_losses = np.array(losses)

        self.is_trained = True

        # Save the model
        dump_loc = self.save_trained_results(output_dir, "dqn_model_" + self.input_type, best_model=best_model)
        print(f"Training results saved at {dump_loc}")

        return
