"""Model 2: censored (Tobit) regression on log10 peak area.

Model 1 threw away most of the signal: an area of 9,000 and an area of 20 both became
"does not fly". This model predicts the area itself and treats the 362 zeros as what they
are -- a measurement that fell below the instrument's floor, not a measurement of zero.

    z_i = log10(area_i) = a + x_i . b + e_i
    e_i ~ Normal(0, s_i),   s_i^2 = s_model^2 + s_meas(level_i)^2

Observed rows contribute the usual Normal density. Censored rows (area = 0) contribute
P(z_i < L) instead, so they inform the fit without pretending to a value.

The two noise terms are kept apart on purpose:

  s_meas   what the instrument does to the same sample twice. Measured, not fitted, from the
           replicate control wells in reactivity_wells.tsv -- different compounds, so this
           does not reuse the training data. It grows as the signal falls (0.033 at log10
           area 5, 0.078 at log10 area 2).
  s_model  everything structure cannot explain. Fitted. This is the part more data could
           shrink; s_meas is not.

The payoff: detection probability comes out at prediction time, for any threshold T, with no
refitting --

    P(flies | x*, T) = Phi( (mu* - log10 T) / sqrt(s_eps^2 + s*^2) )

where s* is the posterior uncertainty in mu* and s_eps the noise above.

Writes to data/processed/:
  model2_noise_prior.json                  s_meas(level) fitted from the replicate wells
  model2_<tag>_<add>_<split>_<prior>_laplace.npz
  model2_<tag>_<add>_<split>_<prior>_mcmc.nc
  model2_summary.csv

    python scripts/fit_model2.py --no-mcmc     # seconds
    python scripts/fit_model2.py               # with the MCMC check, ~25 min
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"

SEED = 20261124
GAUSSIAN_SCALE = 2.5
CUTOFF = 10_000                 # the lab's "flies" threshold, used only to score, never to fit
DRAWS, TUNE, CHAINS = 1000, 1000, 4
TARGET_ACCEPT = 0.95
MIN_SIGMA_MEAS = 0.02           # floor, so the fitted line cannot go negative at high signal


# --------------------------------------------------------------------------- #
# measurement noise, from an independent experiment
# --------------------------------------------------------------------------- #

def noise_prior(cache: bool = True) -> dict:
    """Fit s_meas as a straight line in log10 area, using replicate control wells.

    Each compound's no-enzyme control was run in 3-4 wells on the same instrument, so the
    spread within a group is pure measurement noise. Those are different compounds from the
    Will It Fly set.
    """
    out = PROC / "model2_noise_prior.json"
    if cache and out.exists():
        return json.loads(out.read_text())

    wells = pd.read_csv(RAW / "reactivity_wells.tsv", sep="\t")
    ctl = wells[wells["condition"].str.lower() == "control"]

    rows = []
    for _, s in ctl.groupby(["ocnt_batch", "enzyme"])["area"]:
        pos = s[s > 0]
        if len(pos) >= 3:
            lg = np.log10(pos)
            rows.append({"level": lg.mean(), "sd": lg.std(ddof=1), "n": len(pos)})
    r = pd.DataFrame(rows)

    A = np.vstack([np.ones(len(r)), r["level"]]).T
    (c0, c1), *_ = np.linalg.lstsq(A, r["sd"].to_numpy(), rcond=None)
    pooled = float(np.sqrt(((r["n"] - 1) * r["sd"] ** 2).sum() / (r["n"] - 1).sum()))

    prior = {"intercept": float(c0), "slope": float(c1), "pooled_sd": pooled,
             "n_groups": int(len(r)), "min_sigma": MIN_SIGMA_MEAS}
    out.write_text(json.dumps(prior, indent=2))
    return prior


def sigma_meas(level: np.ndarray, prior: dict) -> np.ndarray:
    """Measurement noise at a given log10 area."""
    return np.clip(prior["intercept"] + prior["slope"] * level, prior["min_sigma"], None)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load(blocks: list[str], additive: str):
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
    corr = X.corr(numeric_only=True).abs().to_numpy().copy()
    np.fill_diagonal(corr, 0.0)
    cols = X.columns.to_list()
    dupes = {cols[j] for i in range(len(cols)) for j in range(i + 1, len(cols)) if corr[i, j] > 0.999}
    X = X.drop(columns=sorted(dupes))

    area = cur[f"area_{additive}"].to_numpy(float)
    censored = area <= 0
    # detection limit: the smallest area the instrument actually reported
    limit = float(np.log10(area[~censored].min()))
    z = np.where(censored, limit, np.log10(np.where(area > 0, area, 1.0)))
    return cur, X, z, censored, limit


# --------------------------------------------------------------------------- #
# Laplace: MAP + Hessian on [a, b, log s_model]
# --------------------------------------------------------------------------- #

def _unpack(w):
    return w[:-1], w[-1]          # coefficients (intercept first), log s_model


def neg_log_post(w, A, z, cens, limit, prec, s_meas2):
    beta, lsm = _unpack(w)
    mu = A @ beta
    var = np.exp(2 * lsm) + s_meas2
    sd = np.sqrt(var)

    obs = ~cens
    ll = np.sum(-0.5 * (z[obs] - mu[obs]) ** 2 / var[obs] - 0.5 * np.log(var[obs]))
    if cens.any():
        u = (limit - mu[cens]) / sd[cens]
        ll += np.sum(stats.norm.logcdf(u))
    return -(ll - 0.5 * np.sum(prec * beta[1:] ** 2) - 0.5 * lsm ** 2)


def grad_neg_log_post(w, A, z, cens, limit, prec, s_meas2):
    beta, lsm = _unpack(w)
    mu = A @ beta
    var = np.exp(2 * lsm) + s_meas2
    sd = np.sqrt(var)
    dvar_dlsm = 2.0 * np.exp(2 * lsm)

    dmu = np.zeros_like(mu)
    dlsm = 0.0

    obs = ~cens
    resid = z[obs] - mu[obs]
    dmu[obs] = resid / var[obs]
    dlsm += np.sum((0.5 * resid ** 2 / var[obs] ** 2 - 0.5 / var[obs]) * dvar_dlsm)

    if cens.any():
        u = (limit - mu[cens]) / sd[cens]
        lam = np.exp(stats.norm.logpdf(u) - stats.norm.logcdf(u))   # inverse Mills ratio
        dmu[cens] = -lam / sd[cens]
        dlsm += np.sum(lam * (-u / (2 * var[cens])) * dvar_dlsm)

    g_beta = A.T @ dmu
    g_beta[1:] -= prec * beta[1:]
    return -np.concatenate([g_beta, [dlsm - lsm]])


def fit_laplace(X, z, cens, limit, prior_type, s_meas2):
    n, d = X.shape
    A = np.hstack([np.ones((n, 1)), X])
    prec = np.full(d, 1.0 / GAUSSIAN_SCALE ** 2)
    w = np.concatenate([np.zeros(d + 1), [np.log(0.3)]])
    w[0] = z[~cens].mean()

    for _ in range(1 if prior_type == "gaussian" else 20):
        res = optimize.minimize(neg_log_post, w, jac=grad_neg_log_post,
                                args=(A, z, cens, limit, prec, s_meas2),
                                method="L-BFGS-B", options={"maxiter": 1000})
        w = res.x
        if prior_type == "ard":
            H = hessian_fd(w, A, z, cens, limit, prec, s_meas2)
            var_b = np.clip(np.diag(np.linalg.inv(H))[1:-1], 1e-12, None)
            new = 1.0 / np.clip(w[1:-1] ** 2 + var_b, 1e-8, None)
            if np.allclose(new, prec, rtol=1e-3):
                prec = new
                break
            prec = new

    H = hessian_fd(w, A, z, cens, limit, prec, s_meas2)
    return {"mean": w, "cov": np.linalg.inv(H), "prec": prec, "limit": limit}


def hessian_fd(w, *args, eps=1e-5):
    """Hessian by finite differences of the analytic gradient."""
    n = len(w)
    H = np.zeros((n, n))
    for i in range(n):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        H[i] = (grad_neg_log_post(wp, *args) - grad_neg_log_post(wm, *args)) / (2 * eps)
    return 0.5 * (H + H.T)


def predict(fit, X, prior, cutoff=CUTOFF, n_draws=4000, seed=SEED):
    """Posterior predictive mean of log10 area, and P(area > cutoff).

    The probability combines what the model does not know about this molecule (the spread of
    mu* across posterior draws) with what the instrument cannot resolve (s_eps), exactly as in
    the proposal's formula.
    """
    rng = np.random.default_rng(seed)
    A = np.hstack([np.ones((len(X), 1)), X])
    cov = fit["cov"]
    L = np.linalg.cholesky(cov + 1e-9 * np.eye(len(cov)))
    W = fit["mean"] + rng.standard_normal((n_draws, len(fit["mean"]))) @ L.T

    mu = A @ W[:, :-1].T                       # n x draws
    s_model = np.exp(W[:, -1])[None, :]
    s_eps = np.sqrt(s_model ** 2 + sigma_meas(mu, prior) ** 2)
    p = stats.norm.sf((np.log10(cutoff) - mu) / s_eps)
    return mu.mean(1), p.mean(1), mu.std(1)


# --------------------------------------------------------------------------- #
# MCMC
# --------------------------------------------------------------------------- #

def fit_mcmc(X, z, cens, limit, prior_type, s_meas2, draws, tune, chains):
    import nutpie
    import pymc as pm
    import pytensor.tensor as pt

    d = X.shape[1]
    obs = ~cens
    with pm.Model() as model:
        a = pm.Normal("a", z[obs].mean(), 2.0)
        if prior_type == "gaussian":
            b = pm.Normal("b", 0.0, GAUSSIAN_SCALE, shape=d)
        else:
            tau = pm.HalfNormal("tau", 1.0, shape=d)
            zb = pm.Normal("zb", 0.0, 1.0, shape=d)
            b = pm.Deterministic("b", zb * tau)
        s_model = pm.HalfNormal("s_model", 1.0)

        mu = a + X @ b
        sd = pt.sqrt(s_model ** 2 + s_meas2)

        pm.Normal("z_obs", mu=mu[obs], sigma=sd[obs], observed=z[obs])
        if cens.any():      # left-censored: only "below the limit" is known
            pm.Potential("z_cens", pm.logcdf(pm.Normal.dist(mu=mu[cens], sigma=sd[cens]), limit))

    return nutpie.sample(nutpie.compile_pymc_model(model), draws=draws, tune=tune,
                         chains=chains, target_accept=TARGET_ACCEPT, seed=SEED,
                         progress_bar=False)


def mcmc_predict(idata, X, prior, cutoff=CUTOFF):
    post = idata.posterior
    b = post["b"].to_numpy().reshape(-1, X.shape[1])
    a = post["a"].to_numpy().reshape(-1)
    s_model = post["s_model"].to_numpy().reshape(-1)
    mu = a[None, :] + X @ b.T
    s_eps = np.sqrt(s_model[None, :] ** 2 + sigma_meas(mu, prior) ** 2)
    p = stats.norm.sf((np.log10(cutoff) - mu) / s_eps)
    return mu.mean(1), p.mean(1)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", nargs="+", default=["chem"])
    ap.add_argument("--additive", default="formate", choices=["formate", "fluoride"])
    ap.add_argument("--splits", nargs="+", default=["split_random", "split_cluster"])
    ap.add_argument("--priors", nargs="+", default=["gaussian", "ard"])
    ap.add_argument("--no-mcmc", action="store_true")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--tune", type=int, default=TUNE)
    ap.add_argument("--chains", type=int, default=CHAINS)
    args = ap.parse_args()

    prior = noise_prior()
    print(f"measurement noise from {prior['n_groups']} replicate groups: "
          f"sigma = {prior['intercept']:.4f} {prior['slope']:+.4f} * log10 area  "
          f"(pooled {prior['pooled_sd']:.4f})")

    cur, Xdf, z, cens, limit = load(args.features, args.additive)
    tag = "+".join(args.features)
    y_flies = cur[f"flies_{args.additive}"].to_numpy()
    print(f"{len(cur)} compounds, {Xdf.shape[1]} features ({tag}), "
          f"{cens.sum()} censored at log10 area {limit:.2f}")

    # noise at each point: from the observed level, or at the limit for censored rows
    s_meas2_all = sigma_meas(z, prior) ** 2

    rows = []
    for split_col in args.splits:
        split = cur[split_col].to_numpy()
        tr, te = split == "train", split == "test"
        Xa = Xdf.to_numpy(float)
        mu_, sd_ = Xa[tr].mean(0), Xa[tr].std(0)
        Xs = (Xa - mu_) / np.where(sd_ > 1e-9, sd_, 1.0)
        short = split_col.replace("split_", "")

        for prior_type in args.priors:
            t = time.time()
            fit = fit_laplace(Xs[tr], z[tr], cens[tr], limit, prior_type, s_meas2_all[tr])
            mu_hat, p_fly, mu_sd = predict(fit, Xs[te], prior)

            obs_te = ~cens[te]
            rmse = float(np.sqrt(np.mean((z[te][obs_te] - mu_hat[obs_te]) ** 2)))
            s = {"RMSE": rmse,
                 "AUC": roc_auc_score(y_flies[te], p_fly),
                 "log loss": log_loss(y_flies[te], np.clip(p_fly, 1e-6, 1 - 1e-6)),
                 "Brier": brier_score_loss(y_flies[te], p_fly),
                 "s_model": float(np.exp(fit["mean"][-1]))}
            rows.append({"prior": prior_type, "posterior": "laplace", "split": short,
                         "features": tag, "seconds": time.time() - t, **s})
            print(f"  {short:8s} {prior_type:8s} laplace  RMSE {rmse:.3f}  AUC {s['AUC']:.3f}  "
                  f"log loss {s['log loss']:.3f}  s_model {s['s_model']:.3f}  ({time.time()-t:.1f}s)")

            np.savez_compressed(
                PROC / f"model2_{tag}_{args.additive}_{short}_{prior_type}_laplace.npz",
                mean=fit["mean"], cov=fit["cov"], prec=fit["prec"], limit=limit,
                test_idx=np.where(te)[0], test_mu=mu_hat, test_mu_sd=mu_sd, test_pflies=p_fly,
                feature_names=np.array(Xdf.columns.to_list()),
            )

            if args.no_mcmc:
                continue

            t = time.time()
            idata = fit_mcmc(Xs[tr], z[tr], cens[tr], limit, prior_type,
                             s_meas2_all[tr], args.draws, args.tune, args.chains)
            mu_m, p_m = mcmc_predict(idata, Xs[te], prior)
            div = int(idata.sample_stats["diverging"].sum())
            rows.append({"prior": prior_type, "posterior": "mcmc", "split": short,
                         "features": tag, "seconds": time.time() - t, "divergences": div,
                         "RMSE": float(np.sqrt(np.mean((z[te][obs_te] - mu_m[obs_te]) ** 2))),
                         "AUC": roc_auc_score(y_flies[te], p_m),
                         "log loss": log_loss(y_flies[te], np.clip(p_m, 1e-6, 1 - 1e-6)),
                         "Brier": brier_score_loss(y_flies[te], p_m)})
            print(f"  {short:8s} {prior_type:8s} mcmc     AUC {rows[-1]['AUC']:.3f}  "
                  f"divergences {div}  ({time.time()-t:.0f}s)")
            idata.to_netcdf(str(PROC / f"model2_{tag}_{args.additive}_{short}_{prior_type}_mcmc.nc"))

    out = PROC / "model2_summary.csv"
    df = pd.DataFrame(rows)
    if out.exists():
        old = pd.read_csv(out)
        keys = ["prior", "posterior", "split", "features"]
        df = pd.concat([old[~old[keys].apply(tuple, axis=1).isin(df[keys].apply(tuple, axis=1))], df])
    df.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
