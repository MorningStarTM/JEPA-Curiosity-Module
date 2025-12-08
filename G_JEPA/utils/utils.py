import numpy as np
import os
from G_JEPA.utils.logger import logger
import matplotlib.pyplot as plt

def save_episode_rewards(rewards, save_dir, filename="episode_rewards.npy"):
    """
    Saves the episode rewards as a .npy file.

    Args:
        rewards (list or np.ndarray): List of episode rewards to save.
        save_dir (str): Directory path where the file will be saved.
        filename (str): Name of the file to save the rewards in. Default is 'episode_rewards.npy'.
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    np.save(save_path, np.array(rewards))
    logger.info(f"Episode rewards saved to {save_path}")


def load_episode_rewards(file_path):
    """
    Loads the episode rewards from a .npy file.

    Args:
        file_path (str): Full path to the .npy file containing saved episode rewards.

    Returns:
        np.ndarray: Loaded array of episode rewards.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return None
    
    rewards = np.load(file_path)
    logger.info(f"Loaded episode rewards from {file_path}")
    return rewards




def moving_average(x, window=50):
    if window <= 1:
        return x
    x = np.asarray(x, dtype=np.float64)
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[window:] - cumsum[:-window]) / window
    # pad at the beginning so length matches
    pad = np.full(window-1, np.nan)
    return np.concatenate([pad, ma])

def plot_training_rewards(
    rewards,
    window=100,
    title="JEPA + ActorCritic Training Reward",
    save_path=None
):
    rewards = np.asarray(rewards, dtype=np.float64)
    episodes = np.arange(1, len(rewards) + 1)

    # Smoothed curve
    smooth = moving_average(rewards, window=window)

    # For a clean trend view: clip extreme spikes only for visualization
    low_p, high_p = np.percentile(rewards, [1, 99])
    clipped = np.clip(rewards, low_p, high_p)
    smooth_clipped = moving_average(clipped, window=window)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 1) Raw reward (full, including spikes)
    ax = axes[0]
    ax.plot(episodes, rewards, linewidth=0.8, alpha=0.6, label="Raw reward")
    ax.set_ylabel("Reward")
    ax.set_title("Raw rewards (full scale)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    # 2) Log-scale reward (good for huge range)
    ax = axes[1]
    # shift by min+eps to avoid log of non-positive values
    eps = 1e-6
    shift = max(0.0, -np.min(rewards) + eps)
    ax.plot(episodes, np.log10(rewards + shift + eps),
            linewidth=0.8, alpha=0.8, label="log10(reward + shift)")
    ax.set_ylabel("log10(reward)")
    ax.set_title("Rewards in log scale (see order-of-magnitude changes)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    # 3) Smoothed trend with outliers clipped
    ax = axes[2]
    ax.plot(episodes, clipped, linewidth=0.5, alpha=0.3,
            label=f"Raw (clipped to [{low_p:.1f}, {high_p:.1f}] percentiles)")
    ax.plot(episodes, smooth_clipped, linewidth=2.0,
            label=f"Smoothed (window={window})", zorder=3)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Smoothed trend (outliers clipped for visibility)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()  