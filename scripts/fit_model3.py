"""Model 3: Gaussian process classifier with a Tanimoto kernel.

Models 1 and 2 are linear in fixed features. A GP instead works from similarity: two molecules
that share substructures should behave alike. The Tanimoto kernel on Morgan fingerprints,

    k(x, x') = amplitude * |x AND x'| / |x OR x'| + bias

is the graph kernel from Ralaivola et al. (2005), which is where this lab's line of work on
chemical kernels starts.

The reason the proposal wants it is not accuracy. It is that a GP's predictive variance grows
as a molecule moves away from the training set, which gives an applicability domain for free
rather than as a bolt-on rule. This script therefore reports, alongside the usual scores,
whether that variance actually tracks error.

Inference is the Laplace approximation for GP classification (Rasmussen & Williams, Algorithm
3.1), and kernel hyperparameters come from maximising the approximate marginal likelihood
(3.2) -- both as named in the proposal.

Writes to data/processed/:
  model3_<add>_<split>.npz    latent mean/variance, probabilities, chosen hyperparameters
  model3_summary.csv

    python scripts/fit_model3.py                      # all four splits
    python scripts/fit_model3.py --splits split_hard  # just the hard one
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.special import expit, log_expit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

JITTER = 1e-6
MAX_NEWTON = 60
TOL = 1e-7


# --------------------------------------------------------------------------- #
# kernel
# --------------------------------------------------------------------------- #

def tanimoto(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tanimoto similarity between every row of A and every row of B (binary fingerprints)."""
    A = A.astype(np.float32)
    B = B.astype(np.float32)
    inter = A @ B.T
    a, b = A.sum(1), B.sum(1)
    union = a[:, None] + b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float64)


# --------------------------------------------------------------------------- #
# Laplace approximation for GP classification  (R&W Algorithm 3.1)
# --------------------------------------------------------------------------- #

def laplace_gp(K: np.ndarray, y: np.ndarray):
    """Newton's method on the posterior over latent f, with y in {0, 1}.

    Returns the mode, the representer weights, the Cholesky factor of B and sqrt(W) -- all of
    which the prediction step needs -- plus the approximate log marginal likelihood.
    """
    n = len(y)
    f = np.zeros(n)
    prev = -np.inf
    for _ in range(MAX_NEWTON):
        pi = expit(f)
        W = np.maximum(pi * (1 - pi), 1e-10)
        sW = np.sqrt(W)
        B = sW[:, None] * K * sW[None, :]
        B[np.diag_indices(n)] += 1.0
        L = np.linalg.cholesky(B)
        b = W * f + (y - pi)
        a = b - sW * cho_solve((L, True), sW * (K @ b))
        f = K @ a
        obj = -0.5 * a @ f + np.sum(y * f + log_expit(-f))
        if abs(obj - prev) < TOL * max(1.0, abs(obj)):
            break
        prev = obj

    pi = expit(f)
    lml = -0.5 * a @ f + np.sum(y * f + log_expit(-f)) - np.sum(np.log(np.diag(L)))
    return {"f": f, "a": y - pi, "L": L, "sW": sW, "lml": float(lml)}


def gp_predict(fit, Ks: np.ndarray, kss: np.ndarray):
    """Latent mean and variance at test points, then the probability (MacKay's approximation).

    `kss` is the prior variance of each test point: with a Tanimoto kernel that is the same for
    every molecule, so all the variation in the posterior variance comes from how much the
    training set constrains that point -- which is exactly the applicability-domain signal.
    """
    mean = Ks.T @ fit["a"]
    v = solve_triangular(fit["L"], fit["sW"][:, None] * Ks, lower=True)
    var = np.maximum(kss - np.einsum("ij,ij->j", v, v), 1e-12)
    # MacKay (R&W 3.25): the logistic sigmoid, matching the likelihood used in the fit.
    # A probit CDF here would be a different link from the one the model was trained with.
    prob = expit(mean / np.sqrt(1.0 + np.pi * var / 8.0))
    return mean, var, prob


# --------------------------------------------------------------------------- #

def fit_split(S_tr, S_te, y_tr, amplitudes, biases, verbose=True):
    """Pick (amplitude, bias) by marginal likelihood, then predict with the winner."""
    best = None
    for amp in amplitudes:
        for bias in biases:
            K = amp * S_tr + bias
            K[np.diag_indices_from(K)] += JITTER
            fit = laplace_gp(K, y_tr)
            if verbose:
                print(f"      amplitude {amp:5.1f}  bias {bias:4.2f}  log marginal {fit['lml']:10.2f}")
            if best is None or fit["lml"] > best[0]["lml"]:
                best = (fit, amp, bias)
    fit, amp, bias = best
    Ks = amp * S_te + bias
    kss = np.full(S_te.shape[1], amp * 1.0 + bias)
    mean, var, prob = gp_predict(fit, Ks, kss)
    return fit, amp, bias, mean, var, prob


def uncertainty_tracks_error(prob, var, y) -> dict:
    """Does high predictive variance actually flag the wrong answers?

    Three readings: correlation between variance and error, and accuracy inside the most and
    least certain fifths. A usable applicability domain needs the confident fifth to be clearly
    better than the unsure one.
    """
    err = np.abs(prob - y)
    order = np.argsort(var)
    k = len(y) // 5
    conf, unsure = order[:k], order[-k:]
    return {
        "corr(var, error)": float(np.corrcoef(var, err)[0, 1]),
        "AUC confident 20%": float(roc_auc_score(y[conf], prob[conf])) if len(set(y[conf])) > 1 else np.nan,
        "AUC unsure 20%": float(roc_auc_score(y[unsure], prob[unsure])) if len(set(y[unsure])) > 1 else np.nan,
        "accuracy confident 20%": float(np.mean((prob[conf] > 0.5) == y[conf])),
        "accuracy unsure 20%": float(np.mean((prob[unsure] > 0.5) == y[unsure])),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--additive", default="formate", choices=["formate", "fluoride"])
    ap.add_argument("--splits", nargs="+",
                    default=["split_random", "split_cluster", "split_scaffold", "split_hard"])
    # Both grids reach past the values the marginal likelihood picks, so a chosen
    # hyperparameter at the edge means the search was too narrow, not that the edge is best.
    ap.add_argument("--amplitudes", nargs="+", type=float,
                    default=[1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
    ap.add_argument("--biases", nargs="+", type=float,
                    default=[0.0, 0.5, 1.0, 2.0, 4.0])
    args = ap.parse_args()

    cur = pd.read_csv(PROC / "wif_curated.csv")
    fp = np.load(PROC / "morgan_2048_r2.npz", allow_pickle=True)["fp"]
    y = cur[f"flies_{args.additive}"].to_numpy()
    print(f"{len(cur)} compounds, {fp.shape[1]}-bit fingerprints, label flies_{args.additive}")

    rows = []
    for split_col in args.splits:
        if split_col not in cur.columns:
            print(f"  {split_col}: missing -- run scripts/add_hard_split.py first")
            continue
        short = split_col.replace("split_", "")
        tr = (cur[split_col] == "train").to_numpy()
        te = ~tr
        print(f"\n  {short}: {tr.sum()} train, {te.sum()} test")

        t = time.time()
        S_tr = tanimoto(fp[tr], fp[tr])
        S_te = tanimoto(fp[tr], fp[te])
        fit, amp, bias, mean, var, prob = fit_split(S_tr, S_te, y[tr], args.amplitudes, args.biases)
        elapsed = time.time() - t

        diag = uncertainty_tracks_error(prob, var, y[te])
        row = {"split": short, "amplitude": amp, "bias": bias, "log marginal": fit["lml"],
               "AUC": roc_auc_score(y[te], prob),
               "log loss": log_loss(y[te], np.clip(prob, 1e-6, 1 - 1e-6)),
               "Brier": brier_score_loss(y[te], prob),
               "seconds": elapsed, **diag}
        rows.append(row)
        print(f"      chosen amplitude {amp}, bias {bias}  ->  AUC {row['AUC']:.3f}  "
              f"log loss {row['log loss']:.3f}  ({elapsed:.0f}s)")
        print(f"      uncertainty: corr with error {diag['corr(var, error)']:+.3f} | "
              f"accuracy {diag['accuracy confident 20%']:.1%} confident vs "
              f"{diag['accuracy unsure 20%']:.1%} unsure")

        np.savez_compressed(
            PROC / f"model3_{args.additive}_{short}.npz",
            test_idx=np.where(te)[0], latent_mean=mean, latent_var=var, prob=prob,
            amplitude=amp, bias=bias, lml=fit["lml"],
        )
        del S_tr, S_te, fit

    out = PROC / "model3_summary.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        df = pd.concat([old[~old["split"].isin(df["split"])], df])
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
