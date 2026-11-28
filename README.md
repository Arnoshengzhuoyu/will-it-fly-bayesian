# Will It Fly? Predicting Echo-MS detectability from chemical structure

CSE 5015 course project · Arno Yu

A mass spectrometer only weighs molecules that carry a charge. In positive-ion mode a compound
has to pick up a proton to be seen at all, and whether it does is a property of its structure:
basic nitrogens help, acids hurt. A compound that gives a clear peak is said to **fly**.

This project predicts flying from structure as a *calibrated probability* — not just a ranking,
but a number a chemist can act on, with uncertainty that is honest about what the model does
not know. Three Bayesian models, one dataset of 11,353 compounds measured on an Echo-MS.

**The headline result is that calibration, not accuracy, is what separates these models.** Every
candidate ranks about equally well (AUC 0.85–0.89). They differ by a factor of three in whether
their stated probabilities are true.

---

## Results

Held-out test set, ammonium formate. Higher is better except for log loss and ECE. Grouped by
feature set, because a model and the features it is given cannot be compared separately.

| Features | Model | AUC | MCC | log loss | ECE |
|---|---|---|---|---|---|
| chemistry (50) | Random forest | 0.846 | 0.511 | 0.486 | — |
| chemistry (50) | **Model 1** — Bayesian logistic regression | 0.868 | 0.577 | 0.455 | **0.028** |
| chemistry (50) | **Model 2** — censored (Tobit) regression | 0.849 | 0.514 | 0.490 | 0.049 |
| Morgan 2048 | Random forest | 0.863 | 0.564 | 0.509 | — |
| Morgan 2048 | Model 1 | 0.862 | 0.550 | 0.468 | 0.035 |
| Morgan 2048 | **Model 3** — GP, Tanimoto kernel | 0.879 | 0.595 | 0.455 | 0.072 |
| Morgan + chemistry | Random forest | 0.885 | 0.593 | 0.472 | 0.107 |
| Morgan + chemistry | **Model 1** | **0.891** | **0.597** | **0.418** | 0.034 |

**ECE** (expected calibration error) is the average gap between the probability a model
announces and the frequency it achieves. 0.028 means that when the model says 70%, the truth is
within about three points. AUC cannot measure this — it only sees ranking, and a forest's vote
share is not a probability at all.

**Like for like, the Bayesian model wins on both counts.** On identical features Model 1 scores
0.891 against the forest's 0.885 while announcing probabilities three times closer to the truth
(ECE 0.034 against 0.107). The gap on ranking is small; the gap on honesty is not.

The grouping matters. Read down a single column and Model 3's 0.879 looks like it beats
Model 1's 0.868 — but those are different feature sets, and the forest rows show Morgan
fingerprints are worth about +0.017 AUC over the chemistry features on their own. Nearly all of
Model 3's apparent lead is the features, not the model.

![cross-validation](figures/06_validation__3-cross-validation.png)

Five-fold cross-validation inside each training set, four different held-out schemes, on the
chemistry features. Pairing fold by fold — which cancels the difficulty the two models share —
**Model 1 has the higher AUC in 20 folds out of 20** (+0.013 ± 0.006) and the lower calibration
error in 19 of 20 (−0.018 ± 0.012). The accuracy margin is small; the calibration margin is
about half the error, consistently. The single-split comparison on Morgan + chemistry above
points the same way and further, but has no error bars.

The honest summary is not *the Bayesian model predicts better*. It is: **it predicts a little
better and it knows what it knows.** For a screen deciding which compounds are worth running,
the second property is the one worth paying for.

---

## Three findings

### 1. The chemistry the model learns is the chemistry we expected

![coefficients](figures/03_model1_logistic__which-chemistry-makes-a-molecule-fly.png)

Model 1's coefficients with 95% credible intervals. Basic nitrogens, lipophilicity and sp³
character push a molecule towards flying; hydrogen-bond donors and carboxylic acids push against
it. That is the textbook account of positive-mode ionisation, recovered from data.

The credible intervals carry information a point estimate cannot. `n_phosphonic_acid` spans
−2.2 to +1.5 — the library holds too few phosphonic acids to say anything about them. A random
forest would report an importance for that feature and give no hint that it is meaningless.

### 2. The GP ranks best and is calibrated worst

Model 3 wins on AUC and MCC and comes last on ECE, by a factor of nearly three. This is the
Laplace approximation for GP classification understating the magnitude of the latent function,
visible as a number: the probabilities sit too close to 0.5 to be taken at face value.

Two things were wrong in the first implementation and are worth recording, because both looked
fine while running:

- the model was **fitted with a logistic likelihood but predicted through a probit link**. The
  mismatch flattered the log loss (0.435 against 0.455) by accidentally compensating for the
  underconfidence. Correcting it makes the score worse and the model honest.
- the kernel's `bias` hyperparameter sat at the **edge of its search grid**, so the marginal
  likelihood had never been allowed to choose. Widening the grid moved it from 0.5 to 4.0 and
  confirmed that value as a true interior peak.

### 3. The applicability-domain claim does not survive contact with the data

![novelty versus error](figures/05_model3_gp__4-link-two-does-novelty-track-error.png)

**How to read this**, right to left. The question it answers is the practical one: *I can only
afford to run some of these compounds — can the model tell me which calls to trust?* At the
right edge nothing is discarded and accuracy is 80%, the grey line. Moving left discards the
predictions the model is least sure of, keeping only the most trusted. **A trust signal that
works makes the line climb**: fewer compounds kept, higher accuracy among those left.

The proposal assumed a GP's predictive variance would be that signal, and would hand us an
applicability domain for free: unfamiliar molecule, larger variance, less trustworthy
prediction. **It does not.** Filtering by GP variance (red) leaves accuracy flat at 80% all the
way across — discarding the "uncertain" nine tenths buys nothing. Correlation between variance
and error is −0.09 to −0.02 across four splits.

The claim has two links and only the first holds. GP variance does track *novelty* — it is a
working stranger-detector. But on this task novelty is not what makes a prediction wrong, and
that is the link that fails. Plausibly the unusual molecules are unusual *because* they carry
an extreme group — a quaternary ammonium always flies, a sulfonic acid never does — which makes
them strange and easy at once.

Something else does work: **distance from 0.5** (blue), which climbs to nearly 100%. When the
model says 95%, it is right; when it says 52%, it is guessing, and it says so. Keeping the most
confident fifth gives about 97% accuracy against 60% for the least confident. So the proposal's
question — *when should I trust this?* — has a good answer; it is simply not the one assumed.

(The jitter at the far left is small numbers: past x = 0.02 only a few dozen compounds remain
and one mistake moves the percentage several points.)

Part of the reason is the dataset. Clustering at 0.4 Tanimoto leaves 7,067 clusters among
11,353 compounds, 4,583 of them singletons. Random, cluster, scaffold and deliberately-hard
splits all land within 0.01 AUC of each other. **This library has no genuinely out-of-domain
corner to fail on**, so it cannot demonstrate an applicability domain either way.

---

## Validation

![y-randomization](figures/06_validation__2-y-randomization.png)

Y-randomization: shuffle the training labels, refit, score against the real test labels, 25
times per split. The shuffled fits average 0.49 — chance — and the luckiest of 25 reaches 0.70,
while the real model scores 0.85–0.87. Four to five standard deviations clear. Whatever the
model learnt is in the labels, not in the plumbing.

The spread matters as much as the mean here: 50 free coefficients fitted to noise can land on a
direction that correlates with the truth by luck, which is why the check compares against the
best of 25 rather than the average.

**Laplace against MCMC.** Models 1 and 2 were fitted both ways — Laplace approximation and NUTS
(nutpie, 4 chains, 1000 draws). Sixteen runs, **zero divergences**, r̂ ≤ 1.006, and predictions
agreeing to within 0.001 AUC. Laplace takes a second where MCMC takes five minutes, so the fast
approximation is used throughout with the sampler as the audit.

**ARD earns its place exactly where there is dead weight to remove.** Against a fixed Gaussian
prior, on the random split:

| Features | Gaussian | ARD | |
|---|---|---|---|
| chemistry (50) | 0.868 / 0.456 | 0.868 / 0.455 | no difference |
| Morgan 2048 | 0.832 / 0.582 | 0.862 / 0.468 | **+0.030 AUC** |
| Morgan + chemistry | 0.854 / 0.574 | **0.891 / 0.418** | **+0.037 AUC** |

(AUC / log loss.) On 50 hand-picked features ARD changes nothing, because none of them are
useless — a fair prior and a learnt one agree. On 2048 fingerprint bits, most of which are
noise for this task, learning a scale per feature is worth more than any other single choice in
the project. Calibration moves even further: ECE 0.100 → 0.034.

This is the proposal's step 3 — *ARD priors instead of stepwise selection* — doing what it was
chosen for. Shrinking a useless coefficient towards zero while leaving its credible interval
wide is not the same as deleting the feature, and it costs one fit rather than hundreds.

---

## Reproducing

Data is not in the repository. It is public, and the download script verifies MD5s:

```bash
python -m venv .venv
.venv/Scripts/activate                   # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
python scripts/download_data.py          # Zenodo record 21450795
```

Then, in order:

```bash
python scripts/featurize.py              # 3 feature blocks, ~6 min
python scripts/add_hard_split.py         # scaffold and hard splits
python scripts/fit_model1.py             # Laplace + MCMC, ~25 min
python scripts/fit_model2.py
python scripts/fit_model3.py             # GP, ~30 min
python scripts/validate.py               # MCC, ECE, Y-randomization, CV
python scripts/export_figures.py
```

Only `featurize.py` needs anything extra: its XenoSite descriptor block reads ONNX weights from
`weights/onnx`, which are not public. The other two feature blocks, and all three models, run
without them. Every notebook is committed **with its outputs**, so all figures and tables can be
read without running anything at all.

### Layout

| Path | What |
|---|---|
| `notebooks/01_data_curation.ipynb` | curation, labels, censoring, the four frozen splits |
| `notebooks/02_baseline_rf.ipynb` | feature cleanup, random forest baselines |
| `notebooks/03_model1_logistic.ipynb` | Model 1, coefficients, Laplace vs MCMC |
| `notebooks/04_model2_censored.ipynb` | Model 2, measurement-noise prior, any-threshold prediction |
| `notebooks/05_model3_gp.ipynb` | Model 3, applicability domain |
| `notebooks/06_validation.ipynb` | comparison table, Y-randomization, cross-validation |
| `scripts/` | the fitting and validation code the notebooks call |
| `figures/` | every figure as a standalone PNG |

---

## Method notes

**Labels.** Peak areas form one smooth distribution with no natural break, so "flies" is a
threshold we choose: area > 10,000, which reproduces the reported pass rates (52.9% formate,
78.6% fluoride). The 362 zeros are *not* measurements of zero — they are "no peak found", and
Model 2 treats them as left-censored rather than pretending to a value.

**Splits.** 20% held out, four ways: random, by cluster (0.4 Tanimoto), by Bemis–Murcko
scaffold, and a deliberately hard split taking the 20% least similar to the rest of the library.
Stereoisomers are kept together throughout, so a pair never straddles train and test.

**Measurement noise (Model 2).** Fitted from 2,444 groups of replicate control wells in the same
data release — different compounds, so no training data is reused. Noise grows as signal falls:
σ = 0.033 at log₁₀ area 5, 0.078 at log₁₀ area 2. Keeping this apart from the model's own error
is what lets the detection probability be recomputed for any threshold without refitting.

**Features.** Three blocks: 50 interpretable chemistry features (SMARTS counts for ionisable
groups plus RDKit descriptors), 41 XenoSite phase1 descriptors from the Swamidass lab, and
2048-bit Morgan fingerprints. The lab descriptors were the weakest set on their own (AUC 0.834)
and added nothing on top of the others — they were built to predict metabolism, not ionisation.

---

## References

1. Zhang H, et al. *Acoustic ejection mass spectrometry for high-throughput analysis.* Anal. Chem. (2021).
2. Simpkins S, et al. *Data release: Building the OpenADMET data engine.* Zenodo 21450795 (2026).
3. Ralaivola L, Swamidass SJ, Saigo H, Baldi P. *Graph kernels for chemical informatics.* Neural Networks 18, 1093–1110 (2005).
4. Rasmussen CE, Williams CKI. *Gaussian Processes for Machine Learning.* MIT Press (2006).
