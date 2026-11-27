"""Model 1: Bayesian logistic regression on the yes/no "flies" label.

    p(y = 1 | x) = sigmoid(a + x . b)

Two priors, so the notebook can show what ARD buys over a plain ridge:

  gaussian   b_j ~ Normal(0, s)          one shared scale, fixed
  ard        b_j ~ Normal(0, tau_j)      a scale per feature, learnt from the data,
                                         so useless features are squeezed towards zero

Two posteriors per prior:

  laplace    MAP fit plus a Gaussian centred there, curvature from the Hessian.
             Seconds. What the proposal uses for the main fit.
  mcmc       nutpie NUTS. Minutes. The check that the Laplace shape is not a lie.

Writes to data/processed/:
  model1_<prior>_laplace.npz    mean, covariance, test predictions
  model1_<prior>_mcmc.nc        full posterior (ArviZ netCDF)
  model1_summary.csv            one row per (prior, posterior, split)

    python scripts/fit_model1.py                      # everything, both splits
    python scripts/fit_model1.py --no-mcmc            # Laplace only, seconds
    python scripts/fit_model1.py --features chem phase1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

SEED = 20261124
GAUSSIAN_SCALE = 2.5        # weakly informative on standardised features
DRAWS, TUNE, CHAINS = 1000, 1000, 4
TARGET_ACCEPT = 0.95        # 0.9 left ~1% divergences on the ARD funnel


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_features(blocks: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Curated labels plus the requested feature blocks, constant columns dropped."""
    cur = pd.read_csv(PROC / "wif_curated.csv")
    parts = []
    for name in blocks:
        if name == "morgan":
            fp = np.load(PROC / "morgan_2048_r2.npz", allow_pickle=True)["fp"]
            parts.append(pd.DataFrame(fp, columns=[f"bit_{i}" for i in range(fp.shape[1])]))
        else:
            d = pd.read_csv(PROC / f"features_{name}.csv")
            assert (d["compound_id"].to_numpy() == cur["compound_id"].to_numpy()).all()
            parts.append(d.drop(columns=["compound_id"]))
    X = pd.concat(parts, axis=1)
    X = X.drop(columns=[c for c in X if X[c].nunique() <= 1])
    # exact duplicates (RDKit and OpenBabel both report MW, HBD, ...)
    corr = X.corr(numeric_only=True).abs().to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    cols = X.columns.to_list()
    dupes = {cols[j] for i in range(len(cols)) for j in range(i + 1, len(cols)) if corr[i, j] > 0.999}
    return cur, X.drop(columns=sorted(dupes))


def standardise(X: np.ndarray, train: np.ndarray) -> np.ndarray:
    """Centre and scale using the training rows only, so the test set stays unseen."""
    mu, sd = X[train].mean(0), X[train].std(0)
    return (X - mu) / np.where(sd > 1e-9, sd, 1.0)


# --------------------------------------------------------------------------- #
# Laplace approximation
# --------------------------------------------------------------------------- #

def fit_laplace(X: np.ndarray, y: np.ndarray, prior: str) -> dict:
    """MAP fit, then a Gaussian posterior from the Hessian at the mode.

    For ARD the scales tau are not sampled; they are set by empirical Bayes
    (alternate between the MAP coefficients and tau_j^2 = b_j^2 + var_j, the
    standard evidence-maximisation update), then the final Hessian is taken
    with tau held fixed. That keeps the posterior Gaussian and closed-form.
    """
    n, d = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])          # intercept first, never penalised

    def neg_log_post(w, prec):
        eta = Xb @ w
        nll = np.sum(np.logaddexp(0.0, eta) - y * eta)
        return nll + 0.5 * np.sum(prec * w[1:] ** 2)

    def grad(w, prec):
        p = special.expit(Xb @ w)
        g = Xb.T @ (p - y)
        g[1:] += prec * w[1:]
        return g

    def hessian(w, prec):
        p = special.expit(Xb @ w)
        H = (Xb * (p * (1 - p))[:, None]).T @ Xb
        H[1:, 1:] += np.diag(prec)
        H[0, 0] += 1.0 / 100.0                     # keep the intercept block invertible
        return H

    prec = np.full(d, 1.0 / GAUSSIAN_SCALE ** 2)
    w = np.zeros(d + 1)
    n_rounds = 1 if prior == "gaussian" else 30

    for _ in range(n_rounds):
        res = optimize.minimize(neg_log_post, w, args=(prec,), jac=grad,
                                method="L-BFGS-B", options={"maxiter": 500})
        w = res.x
        if prior == "ard":
            cov = np.linalg.inv(hessian(w, prec))
            var = np.clip(np.diag(cov)[1:], 1e-12, None)
            tau2 = np.clip(w[1:] ** 2 + var, 1e-8, None)   # evidence update
            new = 1.0 / tau2
            if np.allclose(new, prec, rtol=1e-3):
                prec = new
                break
            prec = new

    cov = np.linalg.inv(hessian(w, prec))
    return {"mean": w, "cov": cov, "prec": prec}


def laplace_predict(fit: dict, X: np.ndarray, n_draws: int = 4000, seed: int = SEED) -> np.ndarray:
    """Posterior predictive probability: average sigmoid over draws from the Gaussian posterior.

    Averaging *after* the sigmoid is the point — it is what makes the prediction
    shrink towards 0.5 when the coefficients are uncertain.
    """
    rng = np.random.default_rng(seed)
    Xb = np.hstack([np.ones((len(X), 1)), X])
    L = np.linalg.cholesky(fit["cov"] + 1e-10 * np.eye(len(fit["cov"])))
    W = fit["mean"] + (rng.standard_normal((n_draws, len(fit["mean"]))) @ L.T)
    return special.expit(Xb @ W.T).mean(axis=1)


# --------------------------------------------------------------------------- #
# MCMC
# --------------------------------------------------------------------------- #

def fit_mcmc(X: np.ndarray, y: np.ndarray, prior: str, draws: int, tune: int, chains: int):
    import nutpie
    import pymc as pm

    d = X.shape[1]
    with pm.Model() as model:
        a = pm.Normal("a", 0.0, 2.0)
        if prior == "gaussian":
            b = pm.Normal("b", 0.0, GAUSSIAN_SCALE, shape=d)
        else:
            # non-centred: sampling z and tau separately avoids the funnel that
            # makes a directly-parameterised ARD model diverge
            tau = pm.HalfNormal("tau", 1.0, shape=d)
            z = pm.Normal("z", 0.0, 1.0, shape=d)
            b = pm.Deterministic("b", z * tau)
        pm.Bernoulli("y", logit_p=a + X @ b, observed=y)

    compiled = nutpie.compile_pymc_model(model)
    return nutpie.sample(compiled, draws=draws, tune=tune, chains=chains,
                         target_accept=TARGET_ACCEPT, seed=SEED, progress_bar=False)


def mcmc_predict(idata, X: np.ndarray) -> np.ndarray:
    post = idata.posterior
    b = post["b"].to_numpy().reshape(-1, X.shape[1])
    a = post["a"].to_numpy().reshape(-1)
    return special.expit(a[:, None] + b @ X.T).mean(axis=0)


# --------------------------------------------------------------------------- #

def scores(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "AUC": roc_auc_score(y, p),
        "log loss": log_loss(y, p),
        "Brier": brier_score_loss(y, p),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", nargs="+", default=["chem"])
    ap.add_argument("--additive", default="formate", choices=["formate", "fluoride"])
    ap.add_argument("--splits", nargs="+", default=["split_random", "split_cluster"])
    ap.add_argument("--priors", nargs="+", default=["gaussian", "ard"])
    ap.add_argument("--no-mcmc", action="store_true", help="Laplace only (seconds)")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--tune", type=int, default=TUNE)
    ap.add_argument("--chains", type=int, default=CHAINS)
    args = ap.parse_args()

    cur, Xdf = load_features(args.features)
    y = cur[f"flies_{args.additive}"].to_numpy()
    tag = "+".join(args.features)
    print(f"{len(cur)} compounds, {Xdf.shape[1]} features ({tag}), label flies_{args.additive}")

    rows = []
    for split_col in args.splits:
        split = cur[split_col].to_numpy()
        tr, te = split == "train", split == "test"
        X = standardise(Xdf.to_numpy(float), tr)
        short = split_col.replace("split_", "")

        for prior in args.priors:
            t = time.time()
            fit = fit_laplace(X[tr], y[tr], prior)
            p = laplace_predict(fit, X[te])
            s = scores(y[te], p)
            rows.append({"prior": prior, "posterior": "laplace", "split": short,
                         "features": tag, "seconds": time.time() - t, **s})
            print(f"  {short:8s} {prior:8s} laplace  AUC {s['AUC']:.3f}  "
                  f"log loss {s['log loss']:.3f}  ({time.time() - t:.1f}s)")

            np.savez_compressed(
                PROC / f"model1_{tag}_{args.additive}_{short}_{prior}_laplace.npz",
                mean=fit["mean"], cov=fit["cov"], prec=fit["prec"],
                test_idx=np.where(te)[0], test_pred=p,
                feature_names=np.array(Xdf.columns.to_list()),
            )

            if args.no_mcmc:
                continue

            t = time.time()
            idata = fit_mcmc(X[tr], y[tr], prior, args.draws, args.tune, args.chains)
            p_m = mcmc_predict(idata, X[te])
            s_m = scores(y[te], p_m)
            div = int(idata.sample_stats["diverging"].sum())
            rows.append({"prior": prior, "posterior": "mcmc", "split": short,
                         "features": tag, "seconds": time.time() - t,
                         "divergences": div, **s_m})
            print(f"  {short:8s} {prior:8s} mcmc     AUC {s_m['AUC']:.3f}  "
                  f"log loss {s_m['log loss']:.3f}  divergences {div}  ({time.time() - t:.0f}s)")

            idata.to_netcdf(str(PROC / f"model1_{tag}_{args.additive}_{short}_{prior}_mcmc.nc"))

    out = PROC / "model1_summary.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        keys = ["prior", "posterior", "split", "features"]
        df = pd.concat([old[~old[keys].apply(tuple, axis=1).isin(df[keys].apply(tuple, axis=1))], df])
    df.to_csv(out, index=False)

    meta = {"features": args.features, "additive": args.additive,
            "n_features": int(Xdf.shape[1]), "gaussian_scale": GAUSSIAN_SCALE,
            "draws": args.draws, "tune": args.tune, "chains": args.chains,
            "target_accept": TARGET_ACCEPT, "seed": SEED}
    (PROC / "model1_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
