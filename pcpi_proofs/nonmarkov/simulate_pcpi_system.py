import numpy as np
import os
import argparse

def simulate_pcpi_system(
    T: int = 120_000,
    burn_in: int = 20_000,
    seed: int = 42,
    alpha: float = 0.02,
    m2: float = 0.5,
    lam: float = 0.3,
    beta: float = 0.35,
    gamma: float = 0.10,
    lambda_M: float = 0.10,
    rho: float = 0.02,
    sigma_eta: float = 0.05,
    sigma_xi: float = 0.02,
):
    """
    Simule le système jouet PCPI (Φ, M, A, Θ) en temps discret.

    Paramètres
    ----------
    T : int
        Longueur totale de la simulation (y compris burn-in).
    burn_in : int
        Nombre de pas de temps à ignorer au début (phase transitoire).
    seed : int
        Graine aléatoire pour reproductibilité.

    Retour
    ------
    dict contenant les séries 'phi', 'M', 'A', 'theta' après burn-in.
    """
    rng = np.random.default_rng(seed)

    # Allocation
    phi = np.zeros(T)
    M = np.zeros(T)
    A = np.zeros(T)
    theta = np.zeros(T)

    # Conditions initiales
    phi[0] = 0.5
    M[0] = 0.0
    A[0] = 0.0
    theta[0] = abs(phi[0] - A[0])

    def dV_dphi(phi_val: float) -> float:
        # Potentiel stable : V(Φ) = 1/2 m^2 Φ^2 + 1/4 λ Φ^4
        return m2 * phi_val + lam * phi_val**3

    for t in range(T - 1):
        # Bruits
        eta_t = rng.normal()
        xi_t = rng.normal()

        # Gradient de potentiel
        grad = dV_dphi(phi[t])

        # Mise à jour de Φ
        phi[t + 1] = (
            phi[t]
            - alpha * grad
            + beta * M[t]
            + gamma * (A[t] - phi[t])
            + sigma_eta * eta_t
        )

        # Mise à jour de M (mémoire)
        M[t + 1] = (1.0 - lambda_M) * M[t] + lambda_M * phi[t]

        # Mise à jour de A (attracteur lent)
        A[t + 1] = A[t] + rho * (phi[t] - A[t]) + sigma_xi * xi_t

        # Tension
        theta[t + 1] = abs(phi[t + 1] - A[t + 1])

    # On enlève la phase de burn-in
    if burn_in >= T:
        raise ValueError("burn_in doit être strictement inférieur à T")
    phi_ss = phi[burn_in:]
    M_ss = M[burn_in:]
    A_ss = A[burn_in:]
    theta_ss = theta[burn_in:]

    return {
        "phi": phi_ss,
        "M": M_ss,
        "A": A_ss,
        "theta": theta_ss,
    }


def main():
    parser = argparse.ArgumentParser(description="Simuler le système jouet PCPI (preuve 1).")
    parser.add_argument("--T", type=int, default=120_000, help="Longueur totale de la simulation")
    parser.add_argument("--burn_in", type=int, default=20_000, help="Burn-in à jeter")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire")
    args = parser.parse_args()

    data = simulate_pcpi_system(T=args.T, burn_in=args.burn_in, seed=args.seed)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    np.savez_compressed(
        os.path.join(data_dir, "pcpi_simulation.npz"),
        phi=data["phi"],
        M=data["M"],
        A=data["A"],
        theta=data["theta"],
    )
    np.save(os.path.join(data_dir, "pcpi_X.npy"), data["phi"])

    print("Simulation terminée.")
    print(f"Séries sauvegardées dans {os.path.join(data_dir, 'pcpi_simulation.npz')}")
    print(f"Série X=phi sauvegardée dans {os.path.join(data_dir, 'pcpi_X.npy')}")
    print(f"Taille de la série stationnaire : {len(data['phi'])} points.")


if __name__ == '__main__':
    main()
