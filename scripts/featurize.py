"""Compute every feature block for the Will It Fly compounds.

Reads  data/processed/wif_curated.csv   (written by notebooks/01_data_curation.ipynb)
Writes data/processed/features_chem.csv    interpretable chemistry, for Model 1
       data/processed/features_phase1.csv  xenosite phase1 descriptors, 41 columns
       data/processed/morgan_2048_r2.npz   Morgan fingerprints, baseline + GP kernel

Rows stay in the order of wif_curated.csv, so every block lines up with the labels
and the frozen splits. Each block is skipped if its output already exists (--force
recomputes).

    python scripts/featurize.py              # everything
    python scripts/featurize.py --limit 50   # quick smoke test
    python scripts/featurize.py --blocks chem morgan
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CURATED = ROOT / "data" / "processed" / "wif_curated.csv"
OUT = ROOT / "data" / "processed"

MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def _setup_env() -> None:
    """Work around the OpenBabel Windows wheel.

    `openbabel/__init__.py` sets BABEL_DATADIR to share/openbabel/<version>, but the
    wheel ships the data files (logp.txt, psa.txt, mr.txt, ...) in bin/data instead,
    so logP / TPSA / MR fail. Setting the variable ourselves is not enough: every
    worker process re-imports openbabel and overwrites it. So copy the files into the
    folder OpenBabel actually looks in, once."""
    import shutil

    import openbabel

    base = Path(openbabel.__file__).parent
    src = base / "bin" / "data"
    dst = base / "share" / "openbabel" / openbabel.__version__
    if src.is_dir() and not (dst / "logp.txt").exists():
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file():
                shutil.copy2(f, dst / f.name)
        print(f"copied OpenBabel data files into {dst}")

    os.environ.setdefault("XENOSITE_MODELS_WEIGHTS", str(ROOT / "weights" / "onnx"))


# --------------------------------------------------------------------------- #
# Block 1: interpretable chemistry (Model 1)
# --------------------------------------------------------------------------- #

# Ionization-relevant groups. In positive-ion mode a molecule flies if it can hold
# an extra proton, so basic nitrogens help and acids hurt.
SMARTS = {
    # basic nitrogen: sp3 N, not amide / sulfonamide / nitro / aromatic
    "basic_N": "[NX3;H2,H1,H0;!$(N[#6]=[O,N,S]);!$(N[S,P]=O);!$(N[+]);!$(N=*);!$(Nn)]",
    "aliphatic_amine_1": "[NX3;H2][CX4]",
    "aliphatic_amine_2": "[NX3;H1]([CX4])[CX4]",
    "aliphatic_amine_3": "[NX3;H0]([CX4])([CX4])[CX4]",
    "aromatic_N_basic": "[nX2;$(n1ccccc1),$(n1cccn1),$(n1ccnc1)]",
    "aniline_N": "[NX3;H2,H1,H0][c]",
    "amidine_guanidine": "[NX3][CX3]=[NX2]",
    "quaternary_N": "[NX4+]",
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "sulfonic_acid": "[SX4](=O)(=O)[OX2H1]",
    "phosphonic_acid": "[PX4](=O)([OX2H1])",
    "tetrazole": "c1nnn[nH]1",
    "acidic_NH": "[NX3;H1](S(=O)=O)",  # sulfonamide NH, weakly acidic
    "phenol": "[OX2H][c]",
    "amide": "[NX3][CX3]=[OX1]",
    "ester": "[CX3](=O)[OX2][#6]",
    "nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",  # RDKit canonicalizes to the charged form
    "halogen": "[F,Cl,Br,I]",
    "sulfur": "[#16]",
}

RDKIT_DESCRIPTORS = [
    "MolWt", "MolLogP", "MolMR", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount", "NumAromaticRings", "NumAliphaticRings",
    "FractionCSP3", "HeavyAtomCount", "NHOHCount", "NOCount", "LabuteASA",
    "BalabanJ", "BertzCT", "Chi0v", "Chi1v", "Chi2v", "Kappa1", "Kappa2", "Kappa3",
    "HallKierAlpha", "MaxPartialCharge", "MinPartialCharge", "NumHeteroatoms",
    "NumSaturatedRings", "qed",
]


def block_chem(smiles: list[str]) -> pd.DataFrame:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    patterns = {k: Chem.MolFromSmarts(v) for k, v in SMARTS.items()}
    bad = [k for k, v in patterns.items() if v is None]
    if bad:
        raise ValueError(f"bad SMARTS: {bad}")

    desc_fn = {n: f for n, f in Descriptors.descList if n in RDKIT_DESCRIPTORS}
    missing = set(RDKIT_DESCRIPTORS) - set(desc_fn)
    if missing:
        raise ValueError(f"RDKit descriptors not found: {sorted(missing)}")

    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        row = {f"n_{k}": len(mol.GetSubstructMatches(p)) for k, p in patterns.items()}
        for name in RDKIT_DESCRIPTORS:
            try:
                row[name] = float(desc_fn[name](mol))
            except Exception:       # MaxPartialCharge fails on a few exotic atoms
                row[name] = np.nan
        # the single most predictive contrast in the proposal's Fig 1B
        row["has_basic_N"] = int(row["n_basic_N"] + row["n_aromatic_N_basic"] > 0)
        row["has_acid"] = int(
            row["n_carboxylic_acid"] + row["n_sulfonic_acid"]
            + row["n_phosphonic_acid"] + row["n_tetrazole"] > 0
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Block 2: Morgan fingerprints (baseline + Tanimoto GP kernel)
# --------------------------------------------------------------------------- #

def block_morgan(smiles: list[str]) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    out = np.zeros((len(smiles), MORGAN_BITS), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        fp = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))
        out[i] = fp
    return out


# --------------------------------------------------------------------------- #
# Block 3: xenosite phase1 descriptors
# --------------------------------------------------------------------------- #
# 16 MolDesc__* (OpenBabel topology) + top-5 site scores for each of 5 phase1 heads.
# NOTE: MolDesc__heavy_atoms counts atoms after hydrogens are added, so it equals
# MolDesc__atoms; this matches the lab's legacy definition, not RDKit's.

PHASE1_HEADS = [
    "stable_oxygenation", "unstable_oxygenation",
    "dehydrogenation", "reduction", "hydrolysis",
]
TOPN = 5


def block_phase1(smiles: list[str], workers: int) -> pd.DataFrame:
    from rdkit import Chem
    from xenosite.predict import predict_many
    from xenosite.predict.v1.features.bond_lonepair import BondAndLonePairTD, phase1_pymol
    from xenosite.predict.v1.features.phase1_mol import _MOL_DESC

    # --- molecule-level OpenBabel descriptors (one row per molecule) ---
    mol_desc = []
    for smi in smiles:
        try:
            rows = BondAndLonePairTD(phase1_pymol(Chem.MolFromSmiles(smi))).run()
            mol_desc.append({c: float(rows[0][c]) for c in _MOL_DESC} if rows else {})
        except Exception:
            mol_desc.append({})
    desc = pd.DataFrame(mol_desc, columns=_MOL_DESC)

    # --- phase1 site scores, pooled to top-5 per head ---
    results = predict_many(smiles, models="phase1", workers=workers)
    score_rows = []
    for mol in results:
        row: dict[str, float] = {}
        by_head = {r.model.split(".", 1)[1]: r for r in mol.results}
        for head in PHASE1_HEADS:
            res = by_head.get(head)
            vals = []
            if res is not None:
                for attr in ("bond", "atom", "score"):
                    v = getattr(res, attr, None)
                    if v:
                        vals = [float(x) for x in v]
                        break
            top = sorted(vals, reverse=True)[:TOPN]
            top += [0.0] * (TOPN - len(top))
            for k, s in enumerate(top):
                row[f"{head}_top{k}"] = s
        score_rows.append(row)
    scores = pd.DataFrame(score_rows)

    return pd.concat([desc, scores], axis=1)


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blocks", nargs="+", default=["chem", "morgan", "phase1"],
                    choices=["chem", "morgan", "phase1"])
    ap.add_argument("--limit", type=int, help="only the first N compounds (smoke test)")
    ap.add_argument("--workers", type=int, default=min(os.cpu_count() or 1, 12))
    ap.add_argument("--force", action="store_true", help="recompute existing outputs")
    ap.add_argument("--suffix", default="", help="append to output filenames")
    args = ap.parse_args()

    _setup_env()
    df = pd.read_csv(CURATED)
    if args.limit:
        df = df.head(args.limit)
    smiles = df["smiles"].tolist()
    ids = df["compound_id"]
    print(f"{len(smiles)} compounds from {CURATED.name}")

    paths = {
        "chem": OUT / f"features_chem{args.suffix}.csv",
        "phase1": OUT / f"features_phase1{args.suffix}.csv",
        "morgan": OUT / f"morgan_{MORGAN_BITS}_r{MORGAN_RADIUS}{args.suffix}.npz",
    }

    for block in args.blocks:
        path = paths[block]
        if path.exists() and not args.force:
            print(f"skip {block}: {path.name} exists (--force to redo)")
            continue
        t = time.time()

        if block == "morgan":
            fp = block_morgan(smiles)
            np.savez_compressed(path, fp=fp, compound_id=ids.to_numpy())
            shape = fp.shape
        else:
            feats = block_chem(smiles) if block == "chem" else block_phase1(smiles, args.workers)
            feats.insert(0, "compound_id", ids.to_numpy())
            feats.to_csv(path, index=False)
            shape = (len(feats), feats.shape[1] - 1)

        print(f"{block:7s} -> {path.name}  {shape[0]} x {shape[1]}  {time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
