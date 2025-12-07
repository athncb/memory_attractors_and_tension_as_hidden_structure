# Memory, Attractors and Tension as Hidden Structure

This repository contains the numerical proofs and code associated with the article:

> **« Mémoire, Attracteurs et Tension comme Structure Cachée des Systèmes Dynamiques »**  
> Athman NCB, 2025  
> Zenodo: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17844456.svg)](https://doi.org/10.5281/zenodo.17844456)

The central claim of the work is that many dynamical models (physics, finance, AI, behavioural systems) implicitly rely on a Markov approximation that ignores a minimal hidden structure:
- a **slow memory** \(M_t\),
- an **internal attractor** \(A_t\),
- and a **tension variable** \(\Theta_t\),

which together capture essential non-Markovian dynamics that are irreducible to any finite-order Markov model on the observable \(\Phi_t\) alone.

This repository makes the three numerical proofs fully reproducible.

---

## Repository structure

```text
memory_attractors_and_tension_as_hidden_structure/
├── LICENSE
├── README.md           # This file
└── pcpi_proofs/
    ├── nonmarkov/      # Synthetic system: violation of Markov assumption
    ├── tension/ # Tension as early-warning signal of regime shift
    └── market/   # Real S&P 500 data: predictive gain with (M, A, Θ)
````

> **Note:** Folder names may differ slightly depending on your local layout
> (`pcpi_proof1`, `pcpi_proof2`, `pcpi_proof3`, etc.). The logic remains the same:
> each subdirectory contains a self-contained experiment.

---

## 1. Numerical Proof #1 — Non-Markovianity on a controlled system

**Goal.**
Show that a system driven by a coherence equation with hidden memory and attractor
((\Phi_t, M_t, A_t, \Theta_t)) violates the Markov assumption when only (\Phi_t) is observed.

**Main results.**

* AR(p) models on (\Phi_t) saturate at a given error level.
* Adding the “PCPI-reduced” variables ((M_t, A_t)) significantly improves prediction.
* A Kolmogorov–Smirnov test on conditional distributions shows a clear violation of the Markov property.

**Typical workflow (from inside the folder, e.g. `pcpi_proofs/proof1_nonmarkov/`):**

```bash
# 1) Simulate the synthetic system
python simulate_pcpi_system.py

# 2) Run Markov vs PCPI analysis
python analysis_markov_vs_pcpi.py
```

This will:

* generate synthetic trajectories in `data/`,
* fit AR(p) baselines,
* compare a Markov model (X_{t+1} \sim X_t) with a PCPI-reduced model
  (X_{t+1} \sim X_t, M_t),
* run a KS test for conditional distributions,
* store metrics and plots in `results/`.

---

## 2. Numerical Proof #2 — Tension as an early-warning signal

**Goal.**
Show that the tension variable (\Theta_t = |\Phi_t - A_t|) behaves as a robust early-warning indicator of regime change in a controlled, piecewise regime system.

**Idea.**

* The system evolves under different regimes (e.g. different attractor positions).
* Before a regime switch, the tension (\Theta_t) increases in a characteristic way.
* Averaging across many simulated crises yields a clear early-warning profile.

**Typical workflow (from `pcpi_proofs/proof2_tension_crisis/`):**

```bash
# 1) Simulate long trajectories with multiple regime changes
python simulate_pcpi_system_proof2.py

# 2) Analyse crisis windows and build tension profiles
python analysis_theta_crisis.py
```

This will:

* produce simulated data with marked regime changes,
* extract crisis windows around change points,
* compute average profiles of (\Theta_t) and volatility,
* store metrics and figures (e.g. mean tension profile before/after crisis) in `results/`.

The key outcome is that (\Theta_t) systematically rises *before* a crisis,
confirming its role as an early-warning signal in the PCPI framework.

---

## 3. Numerical Proof #3 — Predictive gain on S&P 500

**Goal.**
Demonstrate that, on real financial data (S&P 500 index), the hidden structure
((M_t, A_t, \Theta_t)) improves the prediction of strong negative moves compared to a baseline using only past returns.

**Data.**

* Underlying asset: S&P 500 index (`^GSPC` from Yahoo Finance).
* Period: 1990–2024 (can be adjusted via command line arguments).

**Typical workflow (from `pcpi_proofs/proof3_market_sp500/`):**

```bash
# 1) Download and prepare data, compute features
python download_and_prepare_market_data.py --ticker "^GSPC" --start 1990-01-01 --q_neg 0.05

# 2) Train and evaluate baseline vs PCPI-enhanced models
python analysis_pcpi_market.py --ticker "^GSPC" --train_ratio 0.7
```

This pipeline:

* downloads daily price data from Yahoo Finance,
* builds a feature set including returns and PCPI variables ((M_t, A_t, \Theta_t)),
* defines a binary target: “strong negative move” (e.g. bottom 5% quantile of next-day return),
* compares:

  * a baseline model (Markovian: based only on returns),
  * a PCPI-enhanced model (including memory, attractor and tension),
* evaluates them using:

  * **Brier score**,
  * **log-loss**,
  * **AUC (ROC)**.

The article reports **consistent predictive gains** for the PCPI-enhanced model,
indicating that the hidden structure ((M_t, A_t, \Theta_t)) captures
dynamics not accessible to a pure Markovian view.

---

## Dependencies

All experiments are written in Python (3.10+ recommended).
Typical dependencies include:

* `numpy`
* `scipy`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `yfinance`

A simple installation pattern is:

```bash
pip install numpy scipy pandas matplotlib scikit-learn yfinance
```

You can also create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # sous Windows: .venv\Scripts\activate
pip install -r requirements.txt   # si vous ajoutez un fichier de dépendances
```

---

## Reproducibility

Each proof is:

* **self-contained** (simulation + analysis scripts),
* **deterministic** once the random seed is fixed,
* documented in the associated article.

The goal of this repository is to make the claims of the paper **fully checkable**:

1. Non-Markovianity on a controlled synthetic system.
2. Tension as a robust early-warning signal.
3. Empirical predictive gains on real market data.

---

## Citation

If you use this code or reproduce these experiments, please cite:

```text
Athman NECIB, 2025.
« Mémoire, Attracteurs et Tension comme Structure Cachée des Systèmes Dynamiques ».
Zenodo : [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17844456.svg)](https://doi.org/10.5281/zenodo.17844456) 
```

Once a DOI is created for this repository (via Zenodo GitHub integration),
you can also cite the software directly.

---

## License

This project is released under the **MIT License**.
See the [`LICENSE`](LICENSE) file for details.
