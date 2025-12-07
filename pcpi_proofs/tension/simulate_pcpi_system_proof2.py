import numpy as np
import os
import argparse


def simulate_pcpi_system_proof2(
    T: int = 120_000,
    seed: int = 123,
    eps: float = 0.05,
    sigma: float = 0.05,
    gamma: float = 0.02,
    A_star: float = 1.0,
    lambda_s: float = 0.99,
    S_c: float = 0.12,
    shock_sigma: float = 0.1,
):
    """
    Simule le système jouet pour la preuve 2 :
    - Φ_t : état observable
    - A_t : attracteur lent
    - Θ_t = |Φ_t - A_t| : tension
    - S_t : stress (mémoire exponentielle de Θ)
    - r_t ∈ {-1, +1} : régime

    Une crise est déclenchée lorsque S_t dépasse un seuil S_c,
    ce qui provoque un changement de signe de r_t.
    """
    rng = np.random.default_rng(seed)

    phi = np.zeros(T)
    A = np.zeros(T)
    theta = np.zeros(T)
    S = np.zeros(T)
    r = np.ones(T, dtype=int)  # commence en régime +1

    # conditions initiales
    phi[0] = 0.0
    A[0] = 0.0
    theta[0] = abs(phi[0] - A[0])
    S[0] = theta[0]

    for t in range(T - 1):
        # attracteur structurel du régime
        A_star_t = r[t] * A_star

        # mise à jour de l'attracteur lent
        A[t + 1] = A[t] + gamma * (A_star_t - A[t])

        # mise à jour de l'état
        eta_t = rng.normal()
        phi[t + 1] = phi[t] + eps * (A[t] - phi[t]) + sigma * eta_t

        # tension
        theta[t + 1] = abs(phi[t + 1] - A[t + 1])

        # stress
        S[t + 1] = lambda_s * S[t] + (1.0 - lambda_s) * theta[t + 1]

        # règle de crise
        if S[t + 1] > S_c:
            # changement de régime
            r[t + 1] = -r[t]
            # reset du stress
            S[t + 1] = 0.0
            # petit choc sur phi
            phi[t + 1] += shock_sigma * rng.normal()
        else:
            # pas de crise : régime inchangé
            r[t + 1] = r[t]

    return {
        "phi": phi,
        "A": A,
        "theta": theta,
        "S": S,
        "r": r,
    }


def main():
    parser = argparse.ArgumentParser(description="Simuler le système jouet PCPI (preuve 2).")
    parser.add_argument("--T", type=int, default=120_000, help="Longueur de la simulation")
    parser.add_argument("--seed", type=int, default=123, help="Graine aléatoire")
    args = parser.parse_args()

    data = simulate_pcpi_system_proof2(T=args.T, seed=args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(data_dir, "pcpi_proof2_simulation.npz"),
        phi=data["phi"],
        A=data["A"],
        theta=data["theta"],
        S=data["S"],
        r=data["r"],
    )

    print("Simulation terminée.")
    print(f"Séries sauvegardées dans {os.path.join(data_dir, 'pcpi_proof2_simulation.npz')}")
    print(f"Nombre total de points : {len(data['phi'])}")


if __name__ == "__main__":
    main()
