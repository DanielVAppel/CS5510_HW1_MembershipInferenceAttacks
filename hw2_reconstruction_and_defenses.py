# hw1_reconstruction_and_defenses.py
# Q1(a,b,c): reconstruction attack, defenses, metrics, sweeps.
import math
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

# Config
data_path   = "fake_healthcare_dataset_sample100.csv"  # local CSV path
SAVE_FIGS   = True                                     # save plots instead of show()
OUTDIR      = "figs"                                   # where to save plots
COARSE      = False                                     # True: 1,11,21,... ; False: 1..n
N_TRIALS    = 10                                       # trials per parameter value
SEED        = 0                                        # RNG seed for reproducibility

np.random.seed(SEED)
if SAVE_FIGS:
    os.makedirs(OUTDIR, exist_ok=True)

# Load data
try:
    data: pd.DataFrame = pd.read_csv(data_path)
except Exception as e:
    raise RuntimeError(
        f"Could not load dataset from '{data_path}'. "
        f"Original error: {e}"
    )

pub = ["age", "sex", "blood", "admission"]
target = "result"

assert all(col in data.columns for col in pub + [target]), "Unexpected columns in dataset."
n = len(data)
y_true = data[target].astype(int).to_numpy().reshape(-1)

# Predicates
def make_random_predicate(prime: int = 2003) -> Callable[[pd.DataFrame], np.ndarray]:
    desc = np.random.randint(prime, size=len(pub))
    return lambda df: ((df[pub].to_numpy() @ desc) % prime % 2).astype(bool)

def predicates_to_matrix(df_pub: pd.DataFrame, predicates: List[Callable[[pd.DataFrame], np.ndarray]]) -> np.ndarray:
    masks = [pred(df_pub).astype(int) for pred in predicates]  # list of (n,)
    return np.stack(masks, axis=0)  # (k, n)

# Oracles
def execute_subsetsums_exact(predicates: List[Callable[[pd.DataFrame], np.ndarray]]) -> np.ndarray:
    A = predicates_to_matrix(data[pub], predicates)  # (k, n)
    return (A @ y_true).astype(float)                # (k,)

def execute_subsetsums_round(R: int, predicates: List[Callable[[pd.DataFrame], np.ndarray]]) -> np.ndarray:
    exact = execute_subsetsums_exact(predicates)
    return (np.round(exact / R) * R).astype(float)

def execute_subsetsums_noise(sigma: float, predicates: List[Callable[[pd.DataFrame], np.ndarray]]) -> np.ndarray:
    exact = execute_subsetsums_exact(predicates)
    noise = np.random.normal(0.0, sigma, size=exact.shape)
    return (exact + noise).astype(float)

def execute_subsetsums_sample(t: int, predicates: List[Callable[[pd.DataFrame], np.ndarray]]) -> np.ndarray:
    """
    WITH-REPLACEMENT sampling: draw t indices i.i.d. from {0..n-1}.
    For each predicate row, estimate the sum via (A[:, idx] @ y_true[idx]) * (n/t).
    This is an unbiased estimator of the full sum.
    """
    assert 1 <= t
    idx = np.random.randint(0, n, size=t)             # WITH replacement
    A_full = predicates_to_matrix(data[pub], predicates)  # (k, n)
    est = (A_full[:, idx] @ y_true[idx]) * (n / t)    # (k,)
    return est.astype(float)

# Reconstruction (least squares)
def reconstruction_attack(data_pub: pd.DataFrame,
                          predicates: List[Callable[[pd.DataFrame], np.ndarray]],
                          answers: np.ndarray) -> np.ndarray:
    A = predicates_to_matrix(data_pub, predicates).astype(float)
    b = np.asarray(answers, dtype=float).reshape(-1)
    lam = 1e-6
    ATA = A.T @ A
    ATb = A.T @ b
    x_hat = np.linalg.solve(ATA + lam * np.eye(ATA.shape[0]), ATb)
    return (x_hat >= 0.5).astype(int)

# Metrics
def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def attack_success_fraction(y_pred_bin: np.ndarray, y_true_bin: np.ndarray) -> float:
    y_pred_bin = y_pred_bin.reshape(-1).astype(int)
    y_true_bin = y_true_bin.reshape(-1).astype(int)
    return float(np.mean(y_pred_bin == y_true_bin))

def majority_baseline(y_true_bin: np.ndarray) -> float:
    ones = int(np.sum(y_true_bin))
    zeros = len(y_true_bin) - ones
    return max(ones, zeros) / len(y_true_bin)

# Experiment harness
def run_attack_with_oracle(oracle_fn, oracle_param, n_queries: int, n_trials: int = 10) -> Tuple[float, float]:
    answer_rmses, successes = [], []
    for _ in range(n_trials):
        predicates = [make_random_predicate() for _ in range(n_queries)]
        exact = execute_subsetsums_exact(predicates)
        if oracle_fn is execute_subsetsums_round:
            defended = oracle_fn(int(oracle_param), predicates)
        elif oracle_fn is execute_subsetsums_noise:
            defended = oracle_fn(float(oracle_param), predicates)
        elif oracle_fn is execute_subsetsums_sample:
            defended = oracle_fn(int(oracle_param), predicates)
        else:
            defended = oracle_fn(predicates)
        answer_rmses.append(rmse(defended, exact))
        y_hat = reconstruction_attack(data[pub], predicates, defended)
        successes.append(attack_success_fraction(y_hat, y_true))
    return float(np.mean(answer_rmses)), float(np.mean(successes))

def sweep_and_plot(defense_name: str,
                   oracle_fn,
                   param_values: List[int],
                   n_queries: int,
                   n_trials: int = 10):
    rmses, succs = [], []
    for val in param_values:
        avg_rmse, avg_succ = run_attack_with_oracle(oracle_fn, val, n_queries=n_queries, n_trials=n_trials)
        rmses.append(avg_rmse); succs.append(avg_succ)
        print(f"{defense_name} param={val:4d} | RMSE={avg_rmse:.3f} | attack success={avg_succ:.3f}")

    # RMSE plot
    plt.figure()
    plt.plot(param_values, rmses, marker='o')
    plt.xlabel(f"{defense_name} parameter")
    plt.ylabel("Answer RMSE vs exact")
    plt.title(f"Answer accuracy vs {defense_name} parameter (k={n_queries}, trials={n_trials})")
    plt.grid(True)
    if SAVE_FIGS:
        safe = defense_name.split()[0]  # e.g., "R", "sigma", "t"
        plt.savefig(os.path.join(OUTDIR, f"{safe}_rmse_k{n_queries}_tr{n_trials}.png"), bbox_inches="tight", dpi=150)
    else:
        plt.show()

    # Success plot
    plt.figure()
    plt.plot(param_values, succs, marker='o')
    plt.xlabel(f"{defense_name} parameter")
    plt.ylabel("Attack success (fraction correct)")
    plt.title(f"Reconstruction success vs {defense_name} parameter (k={n_queries}, trials={n_trials})")
    plt.grid(True)
    if SAVE_FIGS:
        safe = defense_name.split()[0]
        plt.savefig(os.path.join(OUTDIR, f"{safe}_success_k{n_queries}_tr{n_trials}.png"), bbox_inches="tight", dpi=150)
    else:
        plt.show()

# Main
if __name__ == "__main__":
    k = 2 * n
    print(f"Dataset size n={n}, using k={k} random queries for Q1(a).")
    exact_rmse, exact_succ = run_attack_with_oracle(execute_subsetsums_exact, None, n_queries=k, n_trials=5)
    print(f"[Exact oracle] RMSE={exact_rmse:.3f}, attack success={exact_succ:.3f}, "
          f"majority baseline={majority_baseline(y_true):.3f}")

    # Parameter grid
    params = list(range(1, n+1, 10)) if COARSE else list(range(1, n+1))

    print("\n=== Rounding defense sweep ===")
    sweep_and_plot("R (rounding)", execute_subsetsums_round, params, n_queries=k, n_trials=N_TRIALS)

    print("\n=== Gaussian noise defense sweep ===")
    sweep_and_plot("sigma (noise)", execute_subsetsums_noise, params, n_queries=k, n_trials=N_TRIALS)

    print("\n=== Sampling defense sweep ===")
    sweep_and_plot("t (sampling size)", execute_subsetsums_sample, params, n_queries=k, n_trials=N_TRIALS)

    # Q2
    print("\nQ2(a): O_post = O_prior * (TPR / FPR)")
    print("Q2(b): Even with very large TPR, if FPR isn't tiny, a positive result greatly inflates false membership claims;")
    print("        with small priors, odds are dominated by FPR in O_post = O_prior*(TPR/FPR), so tiny FPR is crucial.")
