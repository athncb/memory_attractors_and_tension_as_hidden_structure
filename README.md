# Memory, Attractors and Tension as Hidden Structure

This repository contains the numerical proofs and code associated with the article:

> **« Mémoire, Attracteurs et Tension comme Structure Cachée des Systèmes Dynamiques »**  
> Athman NCB, 2025  
> Zenodo: TODO_ADD_DOI_HERE

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
    ├── tension_crisis/ # Tension as early-warning signal of regime shift
    └── market_sp500/   # Real S&P 500 data: predictive gain with (M, A, Θ)
