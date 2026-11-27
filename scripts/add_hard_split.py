"""Add a harder held-out set to data/processed/wif_curated.csv.

The existing cluster split barely beats the random one (median nearest-train Tanimoto 0.38 vs
0.42, AUC apart by 0.02) because this library is diverse: clustering at 0.4 similarity leaves
7,067 clusters, 4,583 of them singletons, so "whole clusters held out" still scatters the test
set evenly through chemical space.

Model 3's claim is that a GP's uncertainty grows on unfamiliar molecules. Testing that needs a
test set that is actually unfamiliar. Two further splits:

  split_scaffold  Bemis-Murcko scaffolds: every molecule stripped to its ring system, whole
                  scaffolds held out. The standard "new chemotype" test in QSAR.
  split_hard      the 20% of compounds *least* similar to the rest of the library, by mean
                  Tanimoto to their 10 nearest neighbours. A deliberate worst case, not an
                  unbiased estimate of anything -- it answers "does the model know when it is
                  out of its depth", not "how well does it do in general".

Adds columns in place and prints how much harder each one is.

    python scripts/add_hard_split.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "processed" / "wif_curated.csv"

TEST_FRAC = 0.20
N_FOLDS = 5
N_NEIGHBOURS = 10
SEED = 20261124


def group_split(groups: np.ndarray, strat: np.ndarray, frac: float, rng) -> np.ndarray:
    """Hold out whole groups until `frac` of rows, balanced within each label."""
    g = pd.DataFrame({"g": groups, "y": strat})
    gy = g.groupby("g")["y"].agg(lambda s: s.mode()[0])
    counts = g["g"].value_counts()
    chosen = []
    for y in sorted(gy.unique()):
        ids = gy.index[gy == y].to_numpy().copy()
        rng.shuffle(ids)
        target, n = frac * counts[ids].sum(), 0
        for i in ids:
            if n >= target:
                break
            chosen.append(i)
            n += counts[i]
    return g["g"].isin(chosen).to_numpy()


def group_folds(groups: np.ndarray, k: int, rng) -> np.ndarray:
    ids = pd.unique(groups).copy()
    rng.shuffle(ids)
    fold_of = {gid: i % k for i, gid in enumerate(ids)}
    return np.array([fold_of[gid] for gid in groups])


def main() -> None:
    df = pd.read_csv(CURATED)
    rng = np.random.default_rng(SEED)
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [gen.GetFingerprint(m) for m in mols]
    y = df["flies_formate"].to_numpy()

    # --- scaffold split -----------------------------------------------------
    scaffolds = []
    for m in mols:
        core = MurckoScaffold.GetScaffoldForMol(m)
        smi = Chem.MolToSmiles(core) if core.GetNumAtoms() else ""
        scaffolds.append(smi if smi else "ACYCLIC")
    df["scaffold"] = scaffolds
    n_scaf = df["scaffold"].nunique()
    test_s = group_split(df["scaffold"].to_numpy(), y, TEST_FRAC, rng)
    df["split_scaffold"] = np.where(test_s, "test", "train")

    # --- isolation split: the least typical fifth of the library ------------
    # mean similarity to the 10 nearest neighbours; low = sits on its own
    iso = np.empty(len(fps))
    for i, fp in enumerate(fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, fps))
        sims[i] = -1.0
        iso[i] = np.sort(sims)[-N_NEIGHBOURS:].mean()
    df["isolation"] = iso

    # whole stereo groups move together, ranked by the group's typicality
    grp_iso = df.groupby("skeleton_key")["isolation"].transform("mean")
    order = np.argsort(grp_iso.to_numpy(), kind="stable")
    test_h = np.zeros(len(df), bool)
    test_h[order[: int(TEST_FRAC * len(df))]] = True
    # do not let a skeleton straddle the boundary
    straddle = df.groupby("skeleton_key")["skeleton_key"].transform("size") > 1
    for key in df.loc[straddle, "skeleton_key"].unique():
        rows = (df["skeleton_key"] == key).to_numpy()
        if test_h[rows].any():
            test_h[rows] = True
    df["split_hard"] = np.where(test_h, "test", "train")

    for name, test, grp in [("scaffold", test_s, "scaffold"), ("hard", test_h, "skeleton_key")]:
        f = np.full(len(df), -1)
        f[~test] = group_folds(df.loc[~test, grp].to_numpy(), N_FOLDS, rng)
        df[f"fold_{name}"] = f

    # --- how much harder is each split? -------------------------------------
    print(f"{n_scaf} Bemis-Murcko scaffolds for {len(df)} compounds\n")
    print(f"{'split':10s} {'n test':>7s} {'% flying':>9s} {'median NN sim':>14s} {'% below 0.3':>12s}")
    for col in ["split_random", "split_cluster", "split_scaffold", "split_hard"]:
        te = (df[col] == "test").to_numpy()
        train_fps = [fps[i] for i in np.where(~te)[0]]
        nn = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps))
                       for i in np.where(te)[0]])
        print(f"{col.replace('split_', ''):10s} {te.sum():7d} {y[te].mean():8.1%} "
              f"{np.median(nn):14.3f} {np.mean(nn < 0.3):11.1%}")

    df.to_csv(CURATED, index=False)
    print(f"\nwrote {CURATED}  ({df.shape[1]} columns)")


if __name__ == "__main__":
    main()
