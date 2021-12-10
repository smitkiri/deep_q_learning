from agents import (CartPoleQAgent, CartPoleDqnAgent, CartPoleDoubleDqnAgent,
                    CartPoleDuelingDqnAgent, CartPoleNoisyDqnAgent)
from argparse import ArgumentParser
import sys

Q_LEARNING_PARAMS = {
    "num_steps": 120_000,
    "gamma": 0.99,
    "epsilon_range": (1, 0.1),
    "step_size_range": (0.5, 0.01),
    "buckets": (1, 1, 6, 12)
}

DQN_VECTOR_PARAMS = {
    "input_type": "vector",
    "num_episodes": 10_000,
    "gamma": 0.99,
    "epsilon_range": (1, 0.05),
    "eps_num_steps": 9_000,
    "replay_size": 200_000,
    "replay_prepopulate_steps": 50_000,
    "batch_size": 256,
    "target_update": 10_000,
    "policy_update": 4,
    "learning_rate": 1e-3,
    "step_lr_params": (10_000, 0.9),
    "num_layers": 3,
}

DQN_IMAGE_PARAMS = {
    "input_type": "image",
    "num_episodes": 30_000,
    "gamma": 0.99,
    "epsilon_range": (1, 0.05),
    "eps_num_steps": 28_000,
    "replay_size": 200_000,
    "replay_prepopulate_steps": 50_000,
    "batch_size": 128,
    "target_update": 1000,
    "policy_update": 4,
    "learning_rate": 1e-3,
    "step_lr_params": (25_000, 0.5),
    "center_image": True
}


NOISY_DQN_VECTOR_PARAMS = {
    "input_type": "vector",
    "num_episodes": 2_000,
    "gamma": 0.99,
    "replay_size": 200_000,
    "replay_prepopulate_steps": 50_000,
    "batch_size": 256,
    "target_update": 3_000,
    "policy_update": 4,
    "learning_rate": 1e-3,
    "step_lr_params": (10_000, 0.9)
}

NOISY_DQN_IMAGE_PARAMS = {
    "input_type": "image",
    "num_episodes": 1_000,
    "gamma": 0.99,
    "replay_size": 200_000,
    "replay_prepopulate_steps": 5_000,
    "batch_size": 64,
    "target_update": 1_000,
    "policy_update": 4,
    "learning_rate": 1e-4,
    "step_lr_params": None,
    "center_image": True
}


def parse_arguments(default=False):
    """Parse program arguments"""
    parser = ArgumentParser()

    parser.add_argument("--q_learning", "-q", dest="run_q_learning", action="store_true",
                        help="Run Q-Learning", default=default)
    parser.add_argument("--dqnv", "-dv", dest="run_dqn_vector", action="store_true",
                        help="Run the DQN agent on vector data.", default=default)
    parser.add_argument("--dqni", "-di", dest="run_dqn_image", action="store_true",
                        help="Run the DQN agent on image data.", default=default)
    parser.add_argument("--ddqnv", "-ddv", dest="run_double_dqn_vector", action="store_true",
                        help="Run the Double DQN agent in vector data", default=default)
    parser.add_argument("--ddqni", "-ddi", dest="run_double_dqn_image", action="store_true",
                        help="Run the Double DQN agent in image data", default=default)
    parser.add_argument("--duelingdqnv", "-duedqnv", dest="run_dueling_dqn_vector", action="store_true",
                        help="Run the Dueling DQN agent on vector data.")
    parser.add_argument("--duelingdqni", "-duedqni", dest="run_dueling_dqn_image", action="store_true",
                        help="Run the Dueling DQN agent on image data.")
    parser.add_argument("--noisydqnv", "-ndqnv", dest="run_noisy_dqn_vector", action="store_true",
                        help="Run the Noisy DQN agent on vector data.")
    parser.add_argument("--noisydqni", "-ndqni", dest="run_noisy_dqn_image", action="store_true",
                        help="Run the Noisy DQN agent on image data.")

    args = parser.parse_args()
    return args


def main(args):
    if args.run_q_learning:
        print("Running Q-Learning")

        # Run the Q-Learning Agent
        q_learning_agent = CartPoleQAgent(**Q_LEARNING_PARAMS)
        q_learning_agent.run()

    if args.run_dqn_vector:
        print("\n\nRunning DQN on numerical data")

        # Run the DQN Agent on the numerical inputs
        dqn_agent = CartPoleDqnAgent(**DQN_VECTOR_PARAMS)
        dqn_agent.run()

    if args.run_dqn_image:
        print("\n\nRunning DQN on image data")

        # Run DQN Agent on the image input
        dqn_agent = CartPoleDqnAgent(**DQN_IMAGE_PARAMS)
        dqn_agent.run()

    if args.run_double_dqn_vector:
        print("\n\nRunning Double DQN on vector data")

        # Run the Double DQN Agent on numerical inputs
        ddqn_agent = CartPoleDoubleDqnAgent(**DQN_VECTOR_PARAMS)
        ddqn_agent.run()

    if args.run_double_dqn_image:
        print("\n\nRunning Double DQN on image data")

        # Run the Double DQN Agent on numerical inputs
        ddqn_agent = CartPoleDoubleDqnAgent(**DQN_IMAGE_PARAMS)
        ddqn_agent.run()

    if args.run_dueling_dqn_vector:
        print("\n\nRunning Dueling DQN on vector data")

        # Run the Dueling DQN Agent on numerical inputs
        dueling_dqn_agent = CartPoleDuelingDqnAgent(**DQN_VECTOR_PARAMS)
        dueling_dqn_agent.run()

    if args.run_dueling_dqn_image:
        print("\n\nRunning Dueling DQN on image data")

        # Run the Dueling DQN Agent on numerical inputs
        dueling_dqn_agent = CartPoleDuelingDqnAgent(**DQN_IMAGE_PARAMS)
        dueling_dqn_agent.run()

    if args.run_noisy_dqn_vector:
        print("\n\nRunning Noisy DQN on vector data")

        # Run the Noisy DQN Agent on numerical inputs
        noisy_dqn_agent = CartPoleNoisyDqnAgent(**NOISY_DQN_VECTOR_PARAMS)
        noisy_dqn_agent.run()

    if args.run_noisy_dqn_image:
        print("\n\nRunning Noisy DQN on image data")

        # Run the Noisy DQN Agent on image inputs
        noisy_dqn_agent = CartPoleNoisyDqnAgent(**NOISY_DQN_IMAGE_PARAMS)
        noisy_dqn_agent.run()


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_all = False
    else:
        run_all = True

    program_args = parse_arguments(default=run_all)
    main(program_args)
