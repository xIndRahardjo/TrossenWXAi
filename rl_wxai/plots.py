"""Code to create visualizations."""

from pathlib import Path

import matplotlib.pyplot as plt

from rl_wxai.config import FIGURES_DIR


def plot_training_rewards(rewards: list, title: str = "Training Rewards", save: bool = True) -> None:
    """Plot episode rewards over training steps.

    Args:
        rewards: List of reward values per episode.
        title: Plot title.
        save: If True, saves the figure to reports/figures/.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rewards)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    plt.tight_layout()
    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / f"{title.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()
