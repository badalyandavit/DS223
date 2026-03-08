"""
Compare Epsilon-Greedy and Thompson Sampling on a four-armed bandit.

The assignment fixes the arm means to [1, 2, 3, 4] and uses 20,000 trials.
This implementation follows that setting directly:

- Epsilon-Greedy with epsilon_t = 1 / t
- Thompson Sampling with known observation precision
- Gaussian rewards with unit variance around the fixed arm means
- Simple seed-based caching: if a run for (algorithm, seed) already exists,
  it is loaded from disk instead of being recomputed

Outputs
-------
1. CSV files with columns {Bandit, Reward, Algorithm}
2. Learning-process plots for each algorithm on linear and log scales
3. Cumulative reward and cumulative regret comparison plots
4. Logged cumulative reward and cumulative regret for each algorithm
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger


BANDIT_REWARD = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
N_TRIALS = 20000
N_SEEDS = 20
SEEDS = list(range(N_SEEDS))
PRECISION = 1.0

OUTPUT_DIR = Path("outputs")
CSV_DIR = OUTPUT_DIR / "csv"
PLOT_DIR = OUTPUT_DIR / "plots"
CACHE_DIR = OUTPUT_DIR / "cache"


@dataclass
class BanditStats:
    """
    Store the mean cumulative reward and regret across all seeds.

    Parameters
    ----------
    name : str
        Name of the algorithm.
    mean_cum_rewards : np.ndarray
        Mean cumulative reward at each trial across all seeds.
    mean_cum_regrets : np.ndarray
        Mean cumulative regret at each trial across all seeds.
    """

    name: str
    mean_cum_rewards: np.ndarray
    mean_cum_regrets: np.ndarray


class Bandit(ABC):
    """
    Abstract base class for bandit algorithms.

    The original assignment stub includes the abstract methods below and says
    not to remove anything from the Bandit class. Shared helper methods are
    added after those abstract declarations.
    """

    @abstractmethod
    def __init__(self, p: np.ndarray, n_trials: int, seed: int):
        """Initialize a bandit experiment."""

        self.reward_means = np.asarray(p, dtype=float)
        self.n_trials = n_trials
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.n_arms = len(self.reward_means)

        self.total_reward = 0.0
        self.total_regret = 0.0

        self.cum_rewards: list[float] = []
        self.cum_regrets: list[float] = []
        self.trial_bandits: list[int] = []
        self.trial_rewards: list[float] = []

    @abstractmethod
    def __repr__(self) -> str:
        """Return the algorithm name."""

    @abstractmethod
    def pull(self, t: int | None = None) -> int:
        """Choose which arm to pull."""

    @abstractmethod
    def update(self, chosen: int, reward: float) -> None:
        """Update the algorithm after observing a reward."""

    @abstractmethod
    def experiment(self) -> None:
        """Run the full experiment for one seed."""

    @abstractmethod
    def report(self) -> pd.DataFrame:
        """
        Return one run as a DataFrame with the required CSV columns.

        Final aggregate logging and CSV writing are handled after all seeds are
        processed, but this method is kept because the assignment stub requires
        it.
        """

    def sample_reward(self, chosen: int) -> float:
        """
        Draw a reward from the chosen arm.

        Rewards are Gaussian with known unit variance and fixed means equal to
        BANDIT_REWARD.
        """

        return float(self.rng.normal(loc=self.reward_means[chosen], scale=1.0))

    def record_step(self, chosen: int, reward: float) -> None:
        """
        Store reward and regret information after one trial.

        Regret is computed with respect to the best true mean arm.
        """

        optimal_mean = float(np.max(self.reward_means))
        regret = optimal_mean - self.reward_means[chosen]

        self.total_reward += reward
        self.total_regret += regret

        self.cum_rewards.append(self.total_reward)
        self.cum_regrets.append(self.total_regret)
        self.trial_bandits.append(chosen)
        self.trial_rewards.append(reward)

    def build_trial_frame(self) -> pd.DataFrame:
        """Convert one run into the required trial-level DataFrame."""

        return pd.DataFrame(
            {
                "Bandit": np.asarray(self.trial_bandits, dtype=int),
                "Reward": np.asarray(self.trial_rewards, dtype=float),
                "Algorithm": self.__repr__(),
            }
        )


class EpsilonGreedy(Bandit):
    """
    Epsilon-Greedy algorithm with epsilon_t = 1 / t.

    The estimated value of each arm is updated by the standard incremental mean.
    """

    def __init__(self, p: np.ndarray, n_trials: int, seed: int):
        super().__init__(p=p, n_trials=n_trials, seed=seed)
        self.counts = np.zeros(self.n_arms, dtype=int)
        self.estimates = np.zeros(self.n_arms, dtype=float)
        self.name = "EpsilonGreedy"

    def __repr__(self) -> str:
        return self.name

    def pull(self, t: int | None = None) -> int:
        """
        Choose an arm using a decaying exploration rate.

        Parameters
        ----------
        t : int
            Current trial index starting from 1.
        """

        if t is None:
            raise ValueError("EpsilonGreedy.pull requires the current trial number t")

        epsilon = 1.0 / t
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_arms))
        return int(np.argmax(self.estimates))

    def update(self, chosen: int, reward: float) -> None:
        """Update the running average reward estimate for the chosen arm."""

        self.counts[chosen] += 1
        n = self.counts[chosen]
        self.estimates[chosen] += (reward - self.estimates[chosen]) / n

    def experiment(self) -> None:
        """Run the Epsilon-Greedy experiment for all trials."""

        for t in range(1, self.n_trials + 1):
            chosen = self.pull(t=t)
            reward = self.sample_reward(chosen)
            self.update(chosen, reward)
            self.record_step(chosen, reward)

    def report(self) -> pd.DataFrame:
        """Return the required reward DataFrame for one run."""

        return self.build_trial_frame()


class ThompsonSampling(Bandit):
    """
    Thompson Sampling with Gaussian rewards and known precision.

    Each arm has a Normal prior over its mean. Because the reward precision is
    known, the posterior is updated in closed form after every observation.
    """

    def __init__(self, p: np.ndarray, n_trials: int, seed: int):
        super().__init__(p=p, n_trials=n_trials, seed=seed)
        self.posterior_mean = np.zeros(self.n_arms, dtype=float)
        self.posterior_precision = np.ones(self.n_arms, dtype=float)
        self.name = "ThompsonSampling"

    def __repr__(self) -> str:
        return self.name

    def pull(self, t: int | None = None) -> int:
        """
        Sample one mean from each arm posterior and pick the largest sample.

        The argument t is unused here, but it remains in the method signature
        for interface consistency with the abstract base class.
        """

        sampled_means = self.rng.normal(
            loc=self.posterior_mean,
            scale=1.0 / np.sqrt(self.posterior_precision),
        )
        return int(np.argmax(sampled_means))

    def update(self, chosen: int, reward: float) -> None:
        """
        Update the Normal posterior for the chosen arm.

        The likelihood precision is fixed and known in advance.
        """

        old_precision = self.posterior_precision[chosen]
        new_precision = old_precision + PRECISION

        self.posterior_mean[chosen] = (
            old_precision * self.posterior_mean[chosen] + PRECISION * reward
        ) / new_precision
        self.posterior_precision[chosen] = new_precision

    def experiment(self) -> None:
        """Run the Thompson Sampling experiment for all trials."""

        for _ in range(self.n_trials):
            chosen = self.pull()
            reward = self.sample_reward(chosen)
            self.update(chosen, reward)
            self.record_step(chosen, reward)

    def report(self) -> pd.DataFrame:
        """Return the required reward DataFrame for one run."""

        return self.build_trial_frame()


class Visualization:
    """
    Create and save the figures required by the assignment.

    plot1(): learning process for each algorithm on linear and log scales.
    plot2(): cumulative reward and cumulative regret comparison plots.
    """

    def __init__(self, plot_dir: Path):
        self.plot_dir = Path(plot_dir)
        self.plot_dir.mkdir(parents=True, exist_ok=True)

    def _save_plot(self, filename: str) -> None:
        """Save the current figure and log where it was written."""

        path = self.plot_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        logger.info(f"Saved plot to {path}")

    def plot1(self, stats: BanditStats) -> None:
        """
        Plot the learning process for one algorithm.

        The assignment stub mentions visualizing performance on linear and log
        scales, so this method saves both for each algorithm.
        """

        x = np.arange(1, len(stats.mean_cum_rewards) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(x, stats.mean_cum_rewards)
        plt.xlabel("Trial")
        plt.ylabel("Mean Cumulative Reward")
        plt.title(f"Learning Process: {stats.name} (Linear Scale)")
        self._save_plot(f"{stats.name}_learning_process_linear.png")

        plt.figure(figsize=(10, 6))
        plt.plot(x, np.clip(stats.mean_cum_rewards, 1e-9, None))
        plt.xlabel("Trial")
        plt.ylabel("Mean Cumulative Reward")
        plt.title(f"Learning Process: {stats.name} (Log Scale)")
        plt.yscale("log")
        self._save_plot(f"{stats.name}_learning_process_log.png")

    def plot2(self, epsilon_stats: BanditStats, thompson_stats: BanditStats) -> None:
        """
        Plot the algorithm comparisons required by the assignment stub.

        One figure compares cumulative rewards and another compares cumulative
        regrets.
        """

        x = np.arange(1, len(epsilon_stats.mean_cum_rewards) + 1)

        plt.figure(figsize=(10, 6))
        plt.plot(x, epsilon_stats.mean_cum_rewards, label=epsilon_stats.name)
        plt.plot(x, thompson_stats.mean_cum_rewards, label=thompson_stats.name)
        plt.xlabel("Trial")
        plt.ylabel("Mean Cumulative Reward")
        plt.title("Cumulative Reward Comparison")
        plt.legend()
        self._save_plot("cumulative_reward_comparison.png")

        plt.figure(figsize=(10, 6))
        plt.plot(x, epsilon_stats.mean_cum_regrets, label=epsilon_stats.name)
        plt.plot(x, thompson_stats.mean_cum_regrets, label=thompson_stats.name)
        plt.xlabel("Trial")
        plt.ylabel("Mean Cumulative Regret")
        plt.title("Cumulative Regret Comparison")
        plt.legend()
        self._save_plot("cumulative_regret_comparison.png")


def load_cached_run(algo_name: str, seed: int) -> dict[str, np.ndarray] | None:
    """
    Load a cached run for one algorithm and one seed.

    The cache policy is intentionally simple: if the file exists, use it.
    Otherwise, the experiment must be executed and the results cached.
    """

    cache_path = CACHE_DIR / f"{algo_name}_seed_{seed}.npz"
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as data:
        logger.info(f"Loaded cached results for {algo_name} with seed {seed} from {cache_path}")
        return {
            "cum_rewards": np.asarray(data["cum_rewards"], dtype=float),
            "cum_regrets": np.asarray(data["cum_regrets"], dtype=float),
            "trial_bandits": np.asarray(data["trial_bandits"], dtype=int),
            "trial_rewards": np.asarray(data["trial_rewards"], dtype=float),
        }


def save_cached_run(algo_name: str, seed: int, payload: dict[str, np.ndarray]) -> None:
    """Save one completed run to disk and log that it was cached."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{algo_name}_seed_{seed}.npz"
    np.savez_compressed(
        cache_path,
        cum_rewards=np.asarray(payload["cum_rewards"], dtype=float),
        cum_regrets=np.asarray(payload["cum_regrets"], dtype=float),
        trial_bandits=np.asarray(payload["trial_bandits"], dtype=int),
        trial_rewards=np.asarray(payload["trial_rewards"], dtype=float),
    )
    logger.info(f"Cached results for {algo_name} with seed {seed} at {cache_path}")


def write_trial_csv(algo_name: str, all_trial_frames: list[pd.DataFrame]) -> None:
    """
    Write the required trial-level reward CSV for one algorithm.

    The final file contains all cached or computed runs stacked together with
    the required columns {Bandit, Reward, Algorithm}.
    """

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(all_trial_frames, ignore_index=True)
    csv_path = CSV_DIR / f"{algo_name}.csv"
    combined.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")


def run_one_experiment(algo_class: type[Bandit], seed: int) -> dict[str, np.ndarray | pd.DataFrame]:
    """
    Execute one experiment for one algorithm and one seed.

    Parameters
    ----------
    algo_class : type[Bandit]
        The algorithm class to instantiate.
    seed : int
        Random seed for the run.
    """

    algorithm = algo_class(p=BANDIT_REWARD, n_trials=N_TRIALS, seed=seed)
    algorithm.experiment()

    return {
        "cum_rewards": np.asarray(algorithm.cum_rewards, dtype=float),
        "cum_regrets": np.asarray(algorithm.cum_regrets, dtype=float),
        "trial_bandits": np.asarray(algorithm.trial_bandits, dtype=int),
        "trial_rewards": np.asarray(algorithm.trial_rewards, dtype=float),
        "trial_frame": algorithm.report(),
    }


def run_seeded_experiments(algo_class: type[Bandit]) -> BanditStats:
    """
    Run one algorithm across all seeds and aggregate the results.

    For each seed, the function first checks whether a cached run exists.
    If yes, it is loaded. If not, the experiment is executed and cached.
    """

    algo_name = algo_class.__name__
    all_cum_rewards = []
    all_cum_regrets = []
    all_trial_frames = []

    for seed in SEEDS:
        payload = load_cached_run(algo_name, seed)
        if payload is None:
            payload = run_one_experiment(algo_class, seed)
            save_cached_run(algo_name, seed, payload)
            trial_frame = payload["trial_frame"]
        else:
            trial_frame = pd.DataFrame(
                {
                    "Bandit": payload["trial_bandits"],
                    "Reward": payload["trial_rewards"],
                    "Algorithm": algo_name,
                }
            )

        all_cum_rewards.append(np.asarray(payload["cum_rewards"], dtype=float))
        all_cum_regrets.append(np.asarray(payload["cum_regrets"], dtype=float))
        all_trial_frames.append(trial_frame)

    write_trial_csv(algo_name, all_trial_frames)

    mean_cum_rewards = np.mean(np.vstack(all_cum_rewards), axis=0)
    mean_cum_regrets = np.mean(np.vstack(all_cum_regrets), axis=0)

    logger.info(
        f"{algo_name} mean cumulative reward after {N_TRIALS} trials: {mean_cum_rewards[-1]:.2f}"
    )
    logger.info(
        f"{algo_name} mean cumulative regret after {N_TRIALS} trials: {mean_cum_regrets[-1]:.2f}"
    )

    return BanditStats(
        name=algo_name,
        mean_cum_rewards=mean_cum_rewards,
        mean_cum_regrets=mean_cum_regrets,
    )


def comparison() -> None:
    """Run the full comparison pipeline required by the assignment."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    epsilon_stats = run_seeded_experiments(EpsilonGreedy)
    thompson_stats = run_seeded_experiments(ThompsonSampling)

    visualizer = Visualization(PLOT_DIR)
    visualizer.plot1(epsilon_stats)
    visualizer.plot1(thompson_stats)
    visualizer.plot2(epsilon_stats, thompson_stats)

    logger.info(f"Saved plots to {PLOT_DIR}")
    logger.info(f"Saved CSV files to {CSV_DIR}")


if __name__ == "__main__":
    comparison()
