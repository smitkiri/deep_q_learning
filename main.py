from agents import CartPoleQAgent, CartPoleDqnAgent, CartPoleDoubleDqnAgent
from plotting import plot_lengths_returns
from argparse import ArgumentParser
import sys

Q_LEARNING_PARAMS = {
    "num_steps": 120_000,
    "gamma": 0.99,
    "epsilon_range": (1, 0.1),
    "step_size_range": (0.5, 0.01),
    "buckets": (1, 1, 6, 12)
}

DQN_PARAMS = {
    "num_steps": 50_000,
    "gamma": 0.99,
    "epsilon_range": (1, 0.1),
    "replay_size": 200_000,
    "replay_prepopulate_steps": 50_000,
    "batch_size": 64
}


def parse_arguments(default=False):
    """Parse program arguments"""
    parser = ArgumentParser()

    parser.add_argument("--q_learning", "-q", dest="run_q_learning", action="store_true",
                        help="Run Q-Learning", default=default)
    parser.add_argument("--dqn", "-d", dest="run_dqn", action="store_true",
                        help="Run the DQN agent", default=default)
    parser.add_argument("--ddqn", "-dd", dest="run_double_dqn", action="store_true",
                        help="Run the Double DQN agent", default=default)

    args = parser.parse_args()
    return args


def main(args):
    if args.run_q_learning:
        print("Running Q-Learning")

        # Run the Q-Learning Agent
        q_learning_agent = CartPoleQAgent(**Q_LEARNING_PARAMS)
        q_learning_agent.run()

        plot_lengths_returns(
            returns=q_learning_agent.episode_returns,
            lengths=q_learning_agent.episode_lengths,
            smooth_line=True,
            window_size=100,
            output_file="plots/q_learning_basic.png"
        )

    if args.run_dqn:
        print("\n\nRunning DQN")

        # Run the DQN Agent on the numerical inputs
        dqn_agent = CartPoleDqnAgent(**DQN_PARAMS)
        dqn_agent.run()

        plot_lengths_returns(
            returns=dqn_agent.episode_returns,
            lengths=dqn_agent.episode_lengths,
            smooth_line=True,
            window_size=100,
            output_file="plots/dqn_numerical_basic.png"
        )

    if args.run_double_dqn:
        print("\n\nRunning Double DQN")

        # Run the Double DQN Agent on numerical inputs
        ddqn_agent = CartPoleDoubleDqnAgent(**DQN_PARAMS)
        ddqn_agent.run()

        plot_lengths_returns(
            returns=ddqn_agent.episode_returns,
            lengths=ddqn_agent.episode_lengths,
            smooth_line=True,
            window_size=100,
            output_file="plots/double_dqn_numerical_basic.png"
        )


if __name__ == '__main__':
    if len(sys.argv) > 1:
        run_all = False
    else:
        run_all = True

    program_args = parse_arguments(default=run_all)
    main(program_args)
