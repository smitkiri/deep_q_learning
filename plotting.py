from typing import Sequence, Union
import matplotlib.pyplot as plt
import numpy as np
import os


def _rolling_average(data, *, window_size):
    """Smoothen the 1-d data array using a rollin average.

    Args:
        data: 1-d numpy.array
        window_size: size of the smoothing window

    Returns:
        smooth_data: a 1-d numpy.array with the same size as data
    """
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]


def plot_lengths_returns(returns: Sequence[float], lengths: Sequence[float],
                         smooth_line: bool = False, window_size: int = 100,
                         output_file: Union[str, os.PathLike] = None) -> None:
    """
    Plot the episode lengths and returns

    :param returns: Returns per episode
    :param lengths: Episode lengths
    :param smooth_line: Whether to plot a smoothed line
    :param window_size: Rolling average window size if plotting smooth line
    :param output_file: Output file name if saving the plot. None to display the plot
    :return: None
    """
    returns = np.array(returns)
    lengths = np.array(lengths)

    fig, ax = plt.subplots(2, 1, figsize=(16, 12))

    ax[0].plot(returns)
    ax[0].set_title("Returns")
    ax[0].set_xlabel("Episode Number")

    ax[1].plot(lengths)
    ax[1].set_title("Episode Lengths")
    ax[1].set_xlabel("Episode Number")

    if smooth_line:
        ax[0].plot(_rolling_average(returns, window_size=window_size))
        ax[1].plot(_rolling_average(lengths, window_size=window_size))

    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, bbox_inches="tight")
        print(f"\nPlot saved at {output_file}")
