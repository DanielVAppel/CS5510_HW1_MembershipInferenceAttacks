# hw1_reconstruction_and_defenses.py
# Implements Q1(a,b,c): reconstruction attack, three defenses, metrics, experiments.
# Also prints the Q2 (Bayesian MIA) formulas at the end for convenience.

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple

# ------------------------------
# Config: set this to your CSV path
# ------------------------------
# If you're running locally, point this to the local file path.
# If you're running in this Chat workspace, upload the CSV and set path to '/mnt/data/fake_healthcare_dataset_sample100.csv'
data_path = "fake_healthcare_dataset_sample100.csv"  # CHANGE THIS IF NEEDED

# ------------------------------
# Load data
# ------------------------------
try:
    data: pd.DataFrame = pd.read_csv(data_path)
except Exception as e:
    raise RuntimeError(
        f"Could not load dataset from '{data_path}'. "
        f"Download 'fake_healthcare_dataset_sample100.csv' from the course GitHub and update data_path. "
        f"Original error: {e}"
    )

# names of public identifier columns
pub = ["age", "sex", "blood", "admission"]

# variable to reconstruct (0/1)
target = "result"

# sanity checks
assert all(col in data.columns for col in pub + [target]), "Unexpected columns in dataset."
n = len(data)
y_true = data[target].astype(int).to_numpy().reshape(-1)  # (n,)

# ------------------------------
# Utilities: predicate machinery
# ------------------------------
def make_random_predicate(prime: int = 2003) -> Callable[[pd.DataFrame], np.ndarray]:
    """Returns a (pseudo)random predicate function by hashing public identifiers."""
    desc = np.random.randint(prime, size=len(pub))
    # returns a boolean mask over rows
    return lambda df: ((df[pub].to_numpy() @ desc) % prime % 2).astype(bool)

def predicates_to_matrix(df_pub: pd.DataFrame, predicates: List[Callable[[pd.DataFrame], np.ndarray]]) -> np.ndarray:
    """Stack k boolean masks into an (k x n) design matrix A with 0/1 entries (rows=queries)."""
    masks = [pred(df_pub).astype(int) for pred in predicates]  # list of (n,)
    return np.stack(masks, axis=0)  # (k, n)

# ------------------------------
# Exact query oracle (Equation 1)
# ------------------------------
def execute_subsetsums_exact(predicates: List[Callable[[pd.DataFrame], np.ndarray]]) -> np.ndarray:
    """Exact subset-sum answers: sum_{i: q(i)=1} result_i for each predicate q."""
    A = predicates_to_matrix(data[pub], predicates)  # (k, n)
    ans = A @ y_true  # (k,)
    return ans

# ------------------------------
# Defended query oracles (Q1b)
# ------------------------------
def execute_subsetsums_round(R: int, predicates: List[Callable[[pd.DataFrame], np.ndarray]]) -> np.ndarray:
    exact = execute_subsetsums_exact(predicates).astype(float)
    # round to nearest multiple of R
    return (np.round(exact / R) * R).astype(float)

def execute_subsetsums_noise(sigma: float, predicates: List[Callable[[pd.DataFrame], np.ndarray]]) -> np.ndarray:
    exact = execute_subsetsums_exact(predicates).astype(float)
    noise = np.random.normal(loc=0.0, scale=sigma, size=exact.shape)
    return exact + noise

def execute_subsetsums_sample(t: int, predicates: List[Callable[[pd.DataFrame], np.ndarray]]) -> np.ndarray:
    """Sample t rows uniformly without replacement, compute subset sums on that subsample,
    then scale by n/t to produce an unbiased estimate."""
    assert 1 <= t <= n
    idx = np.random.choice(n, size=t, replace=False)
    df_sub = data.iloc[idx]
    y_sub = df_sub[target].astype(int).to_numpy().reshape(-1)
    A_sub = predicates_to_matrix(df_sub[pub], predicates)  # (k, t)
    # compute on subsample, scale up
    est = (A_sub @ y_sub) * (n / t)
    return est.astype(float)

# ------------------------------
# Q1(a): Reconstruction attack
# ------------------------------
def reconstruction_attack(data_pub: pd.DataFrame,
                          predicates: List[Callable[[pd.DataFrame], np.ndarray]],
                          answers: np.ndarray) -> np.ndarray:
    """Reconstructs the hidden 'result' vector from (possibly noisy) subset-sum answers.

    Strategy: form linear system A x ≈ b, where A_{j,i} = 1 if row i satisfies predicate j, else 0.
    Solve least squares for x_hat and threshold to {0,1}.
    """
    # Build design (k x n) and right-hand side (k,)
    A = predicates_to_matrix(data_pub, predicates).astype(float)
    b = np.asarray(answers, dtype=float).reshape(-1)

    # Solve least squares A x ≈ b
    # Use Tikhonov (ridge) regularization very lightly to stabilize when k≈n or noisy:
    lam = 1e-6
    # x_hat = argmin ||Ax - b||^2 + lam ||x||^2
    # Closed form: x = (A^T A + lam I)^{-1} A^T b
    ATA = A.T @ A
    ATb = A.T @ b
    x_hat = np.linalg.solve(ATA + lam * np.eye(ATA.shape[0]), ATb)

    # Threshold to {0,1}. Use 0.5, but you can adapt (e.g., pick threshold that matches total sum if known).
    x_bin = (x_hat >= 0.5).astype(int)

    return x_bin  # shape (n,)

# ------------------------------
# Metrics
# ------------------------------
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

# ------------------------------
# Experiment harness (Q1c)
# ------------------------------
def run_attack_with_oracle(oracle_fn, oracle_param, n_queries: int, n_trials: int = 10) -> Tuple[float, float]:
    """Returns (avg_answer_rmse, avg_attack_success) across n_trials.
       oracle_fn: one of the execute_subsetsums_* functions
       oracle_param: R, sigma, or t (or None for exact)
    """
    answer_rmses = []
    successes = []

    for _ in range(n_trials):
        # generate k random predicates (k = n_queries)
        predicates = [make_random_predicate() for _ in range(n_queries)]

        # get exact answers for accuracy metric
        exact = execute_subsetsums_exact(predicates)

        # get defended (or exact) answers for the attack
        if oracle_fn is execute_subsetsums_round:
            defended = oracle_fn(int(oracle_param), predicates)
        elif oracle_fn is execute_subsetsums_noise:
            defended = oracle_fn(float(oracle_param), predicates)
        elif oracle_fn is execute_subsetsums_sample:
            defended = oracle_fn(int(oracle_param), predicates)
        else:
            defended = oracle_fn(predicates)  # exact

        # measure answer accuracy vs exact (RMSE)
        answer_rmses.append(rmse(defended, exact))

        # run reconstruction attack on defended answers
        y_hat = reconstruction_attack(data[pub], predicates, defended)

        # measure reconstruction success
        successes.append(attack_success_fraction(y_hat, y_true))

    return float(np.mean(answer_rmses)), float(np.mean(successes))

def sweep_and_plot(defense_name: str,
                   oracle_fn,
                   param_values: List[int],
                   n_queries: int,
                   n_trials: int = 10):
    """Sweeps parameter values; plots average RMSE and average attack success separately."""
    rmses = []
    succs = []

    for val in param_values:
        avg_rmse, avg_succ = run_attack_with_oracle(oracle_fn, val, n_queries=n_queries, n_trials=n_trials)
        rmses.append(avg_rmse)
        succs.append(avg_succ)
        print(f"{defense_name} param={val:4d} | RMSE={avg_rmse:.3f} | attack success={avg_succ:.3f}")

    # Plot per instructions: separate plots, default matplotlib styling, no manual colors
    plt.figure()
    plt.plot(param_values, rmses, marker='o')
    plt.xlabel(f"{defense_name} parameter")
    plt.ylabel("Answer RMSE vs exact")
    plt.title(f"Answer accuracy vs {defense_name} parameter (k={n_queries}, trials={n_trials})")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(param_values, succs, marker='o')
    plt.xlabel(f"{defense_name} parameter")
    plt.ylabel("Attack success (fraction correct)")
    plt.title(f"Reconstruction success vs {defense_name} parameter (k={n_queries}, trials={n_trials})")
    plt.grid(True)
    plt.show()

# ------------------------------
# Main demo (comment/uncomment to run)
# ------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # Q1(a): demonstrate near-perfect reconstruction with 2n random queries (exact oracle)
    k = 2 * n
    print(f"Dataset size n={n}, using k={k} random queries for Q1(a).")
    exact_rmse, exact_succ = run_attack_with_oracle(execute_subsetsums_exact, None, n_queries=k, n_trials=5)
    print(f"[Exact oracle] RMSE={exact_rmse:.3f}, attack success={exact_succ:.3f}, "
          f"majority baseline={majority_baseline(y_true):.3f}")

    # Q1(c): sweeps. Full 1..n sweeps are expensive; start with a coarse grid to visualize the transition.
    # You can switch to list(range(1, n+1)) to run the full sweep before making final plots for your report.

    coarse = list(range(1, n+1, max(1, n//10)))  # ~10 points across [1, n]

    print("\n=== Rounding defense sweep ===")
    sweep_and_plot("R (rounding)", execute_subsetsums_round, coarse, n_queries=k, n_trials=10)

    print("\n=== Gaussian noise defense sweep ===")
    sweep_and_plot("sigma (noise)", execute_subsetsums_noise, coarse, n_queries=k, n_trials=10)

    print("\n=== Sampling defense sweep ===")
    sweep_and_plot("t (sampling size)", execute_subsetsums_sample, coarse, n_queries=k, n_trials=10)

    # Q2 (printed for convenience)
    print("\nQ2(a): O_post = O_prior * (TPR / FPR)")
    print("Q2(b): Even with very large TPR, if FPR isn't tiny, a positive result greatly inflates false membership claims;")
    print("        with small priors, odds are dominated by FPR in O_post = O_prior*(TPR/FPR), so tiny FPR is crucial.")
