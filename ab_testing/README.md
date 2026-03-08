# Bandit A/B Comparison

This repository implements the assignment experiment directly:

- four bandits with true mean rewards `[1, 2, 3, 4]`
- `20000` trials per run
- two algorithms:
  - `EpsilonGreedy` with decaying exploration `epsilon_t = 1 / t`
  - `ThompsonSampling` with known precision
- Gaussian rewards with unit variance around the fixed arm means
- repeated runs over multiple seeds
- simple seed-based caching: if a run for a seed already exists, it is loaded instead of recomputed

## Files

```text
.
├── Bandit.py
├── README.md
├── Homework 2.pdf
└── outputs/
    ├── cache/
    ├── csv/
    └── plots/
```

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Run the experiment

```bash
python3 Bandit.py
```

The script uses the constants defined at the top of `Bandit.py`:

- `BANDIT_REWARD = [1, 2, 3, 4]`
- `N_TRIALS = 20000`
- `N_SEEDS = 20`
- `SEEDS = list(range(N_SEEDS))`
- `PRECISION = 1.0`

If you want to change the number of seeds or the precision, edit those constants directly.

## What the script does

For each algorithm and for each seed:

1. check whether a cached result already exists
2. load it if it exists
3. otherwise run the experiment and cache the results

After all seeds are processed, the script:

1. computes the mean cumulative reward across seeds
2. computes the mean cumulative regret across seeds
3. saves the reward data to CSV
4. creates the required plots
5. logs the final cumulative reward and regret

## Outputs

### CSV files

The script creates:

- `outputs/csv/EpsilonGreedy.csv`
- `outputs/csv/ThompsonSampling.csv`

Each CSV contains the required columns:

- `Bandit`
- `Reward`
- `Algorithm`

### Cache files

The script stores one cache file per algorithm and per seed, for example:

- `outputs/cache/EpsilonGreedy_seed_0.npz`
- `outputs/cache/ThompsonSampling_seed_0.npz`

The cache behavior is simple:

- if the file exists, it is reused
- if the file does not exist, the experiment is executed and the results are cached

### Plots

The script creates:

- `outputs/plots/EpsilonGreedy_learning_process_linear.png`
- `outputs/plots/EpsilonGreedy_learning_process_log.png`
- `outputs/plots/ThompsonSampling_learning_process_linear.png`
- `outputs/plots/ThompsonSampling_learning_process_log.png`
- `outputs/plots/cumulative_reward_comparison.png`
- `outputs/plots/cumulative_regret_comparison.png`

## Regret definition

Cumulative regret is computed using the true best arm mean.
Since the best arm has mean `4`, the regret at each step is:

```text
4 - mean_of_chosen_arm
```
