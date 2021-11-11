from agents import CartPoleQAgent
from plotting import plot_lengths_returns

if __name__ == '__main__':

    q_learning_params = {
        "num_steps": 120_000,
        "gamma": 0.99,
        "max_epsilon": 1,
        "min_epsilon": 0.1,
        "max_step_size": 0.5,
        "min_step_size": 0.01,
        "buckets": (1, 1, 6, 12)
    }

    q_learning_agent = CartPoleQAgent(**q_learning_params)
    q_learning_agent.run()

    plot_lengths_returns(
        returns=q_learning_agent.episode_returns,
        lengths=q_learning_agent.episode_lengths,
        smooth_line=True,
        window_size=100,
        output_file="plots/q_learning_basic.png"
    )
