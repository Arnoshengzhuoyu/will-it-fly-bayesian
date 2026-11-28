"""Step 6 of the QSAR workflow: the validation the proposal asks for and the fits skipped.

Three things, none of which the per-model scripts do:

  metrics   MCC and expected calibration error alongside the AUC / log loss / Brier already
            saved, so the comparison table has every column the proposal names. Read from the
            saved test predictions -- nothing is refitted.

  yrand     Y-randomization. Shuffle the labels, refit, score. A model that still scores well
            on scrambled labels is reading something other than chemistry -- a leak between
            train and test, or a feature that encodes the answer. The real AUC has to sit far
            outside the shuffled distribution. OECD guidance makes this a required check.

  cv        5-fold cross-validation inside each training set, which turns a single test-set
            number into a mean and a spread. Without it there is no way to say whether
            "Model 1 beats the forest, 0.868 to 0.846" is a real gap or one draw of noise.
            Model 3 is left out on purpose: one GP fit is ~8 minutes, so five folds across four
            splits would cost hours to answer a question the fast models already settle.

    python scripts/validate.py                     # all three, ~5 minutes
    python scripts/validate.py --what metrics      # instant, no refitting
    python scripts/validate.py --yrand-reps 50     # a tighter null distribution
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (brier_score_loss, log_loss, matthews_corrcoef,
                             roc_auc_score)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
SEED = 20261124
SPLITS = ["random", "cluster", "scaffold", "hard"]


def _load(name: str):
    """Import a sibling script by path, so this works however the repo is laid out."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Mean gap between predicted probability and observed frequency, over equal-mass bins.

    Equal-mass rather than equal-width: with most predictions near 0 or 1, fixed-width bins
    leave the middle ones nearly empty and the number swings on a handful of points.
    """
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    idx = np.digitize(p, edges[1:-1])
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / len(y))


def score(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"AUC": roc_auc_score(y, p),
            "MCC": matthews_corrcoef(y, (p > 0.5).astype(int)),
            "log loss": log_loss(y, p),
            "Brier": brier_score_loss(y, p),
            "ECE": expected_calibration_error(y, p)}


def metrics_table(cur: pd.DataFrame, additive: str) -> pd.DataFrame:
    """Every saved test prediction, scored the same way."""
    y_all = cur[f"flies_{additive}"].to_numpy()
    rows = []

    for f in sorted(PROC.glob(f"model1_*_{additive}_*_laplace.npz")):
        _, feats, _, split, prior, _ = f.stem.split("_")
        z = np.load(f, allow_pickle=True)
        rows.append({"model": "Model 1 logistic", "variant": f"{prior}/{feats}", "split": split,
                     **score(y_all[z["test_idx"]], z["test_pred"])})

    for f in sorted(PROC.glob(f"model2_*_{additive}_*_laplace.npz")):
        _, feats, _, split, prior, _ = f.stem.split("_")
        z = np.load(f, allow_pickle=True)
        rows.append({"model": "Model 2 censored", "variant": f"{prior}/{feats}", "split": split,
                     **score(y_all[z["test_idx"]], z["test_pflies"])})

    for f in sorted(PROC.glob(f"model3_{additive}_*.npz")):
        split = f.stem.split("_")[-1]
        z = np.load(f, allow_pickle=True)
        rows.append({"model": "Model 3 GP", "variant": "tanimoto/morgan", "split": split,
                     **score(y_all[z["test_idx"]], z["prob"])})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Y-randomization
# --------------------------------------------------------------------------- #

def y_randomization(cur, Xdf, y, additive, reps, splits):
    """Refit on shuffled labels. Shuffling happens inside the training set only, so the test
    set keeps its real labels -- the question is whether the *fitted model* can find signal
    that is not there, not whether scrambling both sides destroys the correlation."""
    m1 = _load("fit_model1")
    X_all = Xdf.to_numpy(float)
    rng = np.random.default_rng(SEED)
    rows = []

    for split in splits:
        col = f"split_{split}"
        tr, te = (cur[col] == "train").to_numpy(), (cur[col] == "test").to_numpy()
        X = m1.standardise(X_all, tr)

        real = m1.laplace_predict(m1.fit_laplace(X[tr], y[tr], "ard"), X[te])
        real_auc = roc_auc_score(y[te], real)

        aucs = []
        for _ in range(reps):
            y_shuf = y.copy()
            y_shuf[tr] = rng.permutation(y[tr])
            p = m1.laplace_predict(m1.fit_laplace(X[tr], y_shuf[tr], "ard"), X[te])
            aucs.append(roc_auc_score(y[te], p))

        a = np.array(aucs)
        rows.append({"split": split, "real AUC": real_auc,
                     "shuffled mean": a.mean(), "shuffled sd": a.std(),
                     "shuffled max": a.max(),
                     "z": (real_auc - a.mean()) / max(a.std(), 1e-9), "reps": reps})
        print(f"  {split:9s} real {real_auc:.3f} | shuffled {a.mean():.3f} +- {a.std():.3f} "
              f"(max {a.max():.3f}) | z = {rows[-1]['z']:.0f}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# cross-validation
# --------------------------------------------------------------------------- #

def cross_validate(cur, Xdf, y, additive, splits, tag="chem"):
    """5-fold CV inside each training set. The folds were built with the curated file and keep
    whole scaffolds or clusters together, so a fold is as honest as the outer test set."""
    m1 = _load("fit_model1")
    m2 = _load("fit_model2")
    noise = m2.noise_prior()

    X_all = Xdf.to_numpy(float)
    area = cur[f"area_{additive}"].to_numpy(float)
    cens = area <= 0
    limit = float(np.log10(area[~cens].min()))
    z_obs = np.where(cens, limit, np.log10(np.where(area > 0, area, 1.0)))
    s_meas2 = m2.sigma_meas(z_obs, noise) ** 2

    rows = []
    for split in splits:
        folds = cur[f"fold_{split}"].to_numpy()
        for k in sorted(set(folds[folds >= 0])):
            t0 = time.time()
            tr, va = (folds >= 0) & (folds != k), folds == k
            X = m1.standardise(X_all, tr)

            p1 = m1.laplace_predict(m1.fit_laplace(X[tr], y[tr], "ard"), X[va])
            rows.append({"model": "Model 1 logistic", "features": tag, "split": split,
                         "fold": int(k), **score(y[va], p1)})

            fit2 = m2.fit_laplace(X[tr], z_obs[tr], cens[tr], limit, "ard", s_meas2[tr])
            _, p2, _ = m2.predict(fit2, X[va], noise)
            rows.append({"model": "Model 2 censored", "features": tag, "split": split,
                         "fold": int(k), **score(y[va], p2)})

            rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                        n_jobs=-1, random_state=SEED).fit(X[tr], y[tr])
            rows.append({"model": "RF baseline", "features": tag, "split": split,
                         "fold": int(k), **score(y[va], rf.predict_proba(X[va])[:, 1])})
            print(f"  {split:9s} fold {k}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--what", nargs="+", default=["metrics", "yrand", "cv"],
                    choices=["metrics", "yrand", "cv"])
    ap.add_argument("--additive", default="formate", choices=["formate", "fluoride"])
    ap.add_argument("--features", nargs="+", default=["chem"])
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    ap.add_argument("--yrand-reps", type=int, default=25)
    args = ap.parse_args()

    m1 = _load("fit_model1")
    cur, Xdf = m1.load_features(args.features)
    y = cur[f"flies_{args.additive}"].to_numpy()
    print(f"{len(cur)} compounds, {Xdf.shape[1]} features, label flies_{args.additive}\n")

    if "metrics" in args.what:
        t = time.time()
        tab = metrics_table(cur, args.additive)
        tab.to_csv(PROC / "validation_metrics.csv", index=False)
        print("metrics (every saved prediction, all five measures)")
        print(tab.round(3).to_string(index=False))
        print(f"  -> validation_metrics.csv  ({time.time() - t:.1f}s)\n")

    if "yrand" in args.what:
        t = time.time()
        print(f"Y-randomization, Model 1 ARD, {args.yrand_reps} shuffles per split")
        yr = y_randomization(cur, Xdf, y, args.additive, args.yrand_reps, args.splits)
        yr.to_csv(PROC / "validation_yrandomization.csv", index=False)
        print(f"  -> validation_yrandomization.csv  ({time.time() - t:.0f}s)\n")

    if "cv" in args.what:
        t = time.time()
        print("5-fold cross-validation (Model 1, Model 2, RF baseline)")
        cv = cross_validate(cur, Xdf, y, args.additive, args.splits, tag="+".join(args.features))
        out = PROC / "validation_cv.csv"
        if out.exists():          # keep results for other feature sets
            old = pd.read_csv(out)
            if "features" in old:
                keep = ~old[["features", "split"]].apply(tuple, axis=1).isin(
                    cv[["features", "split"]].apply(tuple, axis=1))
                cv = pd.concat([old[keep], cv], ignore_index=True)
        cv.to_csv(out, index=False)
        agg = (cv.groupby(["split", "model"])[["AUC", "MCC", "log loss", "ECE"]]
                 .agg(["mean", "std"]).round(3))
        print(agg.to_string())
        print(f"  -> validation_cv.csv  ({time.time() - t:.0f}s)")


if __name__ == "__main__":
    main()
