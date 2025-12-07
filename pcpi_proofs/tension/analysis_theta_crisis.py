import os
import json
import numpy as np
from scipy import stats


def detect_crises(r):
    """
    Détecte les instants de crise comme les changements de signe de r_t.
    Retourne un tableau d'indices t_c.
    """
    r = np.asarray(r)
    changes = np.where(r[1:] * r[:-1] < 0)[0] + 1  # t où r_t != r_{t-1}
    return changes


def build_windows(indices, T, half_window):
    """
    Filtre les indices pour lesquels on peut construire une fenêtre [t - half_window, t + half_window].
    Retourne les indices valides.
    """
    valid = [t for t in indices if (t - half_window >= 0 and t + half_window < T)]
    return np.array(valid, dtype=int)


def profile_around_events(series, event_indices, half_window):
    """
    Calcule le profil moyen de 'series' autour des événements.
    Retourne (profile_mean, profile_std).
    """
    series = np.asarray(series)
    T = len(series)
    W = half_window
    windows = []

    for t_c in event_indices:
        if t_c - W < 0 or t_c + W >= T:
            continue
        window = series[t_c - W : t_c + W + 1]
        windows.append(window)

    windows = np.array(windows)
    mean_profile = windows.mean(axis=0)
    std_profile = windows.std(axis=0)
    return mean_profile, std_profile, windows.shape[0]


def build_baseline_mask(T, crisis_indices, half_window):
    """
    Construit un masque booléen des instants 'loin des crises' :
    - distance > half_window de tout t_c.
    """
    mask = np.ones(T, dtype=bool)
    for t_c in crisis_indices:
        start = max(0, t_c - half_window)
        end = min(T, t_c + half_window + 1)
        mask[start:end] = False
    return mask


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    data_path = os.path.join(data_dir, "pcpi_proof2_simulation.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Fichier de simulation introuvable : {data_path}\n"
            "Veuillez d'abord exécuter simulate_pcpi_system_proof2.py"
        )

    data = np.load(data_path)
    phi = data["phi"]
    A = data["A"]
    theta = data["theta"]
    S = data["S"]
    r = data["r"]

    T = len(phi)
    half_window = 80  # taille de fenêtre avant/après crise

    # Volatilité naive
    vol = np.zeros_like(phi)
    vol[1:] = np.abs(phi[1:] - phi[:-1])

    # 1. Détecter les crises
    crisis_indices = detect_crises(r)
    crisis_indices = build_windows(crisis_indices, T, half_window)

    if len(crisis_indices) == 0:
        raise RuntimeError("Aucune crise détectée avec les paramètres actuels.")

    # 2. Profils moyens autour des crises
    theta_profile_mean, theta_profile_std, n_crises_used = profile_around_events(
        theta, crisis_indices, half_window
    )
    vol_profile_mean, vol_profile_std, _ = profile_around_events(
        vol, crisis_indices, half_window
    )

    # 3. Baseline loin des crises
    baseline_mask = build_baseline_mask(T, crisis_indices, half_window)
    theta_baseline = theta[baseline_mask]
    vol_baseline = vol[baseline_mask]

    # 4. Statistiques pré-crise vs baseline pour Θ
    # Fenêtre pré-crise : [t_c - half_window, t_c - 1]
    W = half_window
    pre_theta_values = []
    pre_vol_values = []

    for t_c in crisis_indices:
        pre_slice = slice(t_c - W, t_c)  # t_c non inclus
        pre_theta_values.append(theta[pre_slice])
        pre_vol_values.append(vol[pre_slice])

    pre_theta_values = np.concatenate(pre_theta_values)
    pre_vol_values = np.concatenate(pre_vol_values)

    # Stats pour Θ
    theta_baseline_mean = float(theta_baseline.mean())
    theta_baseline_std = float(theta_baseline.std())
    pre_theta_mean = float(pre_theta_values.mean())
    pre_theta_std = float(pre_theta_values.std())

    # t-test Welch
    t_theta, p_theta = stats.ttest_ind(pre_theta_values, theta_baseline, equal_var=False)
    # taille d'effet (Cohen's d)
    pooled_std_theta = np.sqrt(
        0.5 * (pre_theta_std**2 + theta_baseline_std**2)
    )
    d_theta = float((pre_theta_mean - theta_baseline_mean) / pooled_std_theta)

    # Stats pour V
    vol_baseline_mean = float(vol_baseline.mean())
    vol_baseline_std = float(vol_baseline.std())
    pre_vol_mean = float(pre_vol_values.mean())
    pre_vol_std = float(pre_vol_values.std())

    t_vol, p_vol = stats.ttest_ind(pre_vol_values, vol_baseline, equal_var=False)
    pooled_std_vol = np.sqrt(
        0.5 * (pre_vol_std**2 + vol_baseline_std**2)
    )
    d_vol = float((pre_vol_mean - vol_baseline_mean) / pooled_std_vol)

    # Sauvegarde des profils
    np.save(os.path.join(results_dir, "theta_profile_mean.npy"), theta_profile_mean)
    np.save(os.path.join(results_dir, "theta_profile_std.npy"), theta_profile_std)
    np.save(os.path.join(results_dir, "vol_profile_mean.npy"), vol_profile_mean)
    np.save(os.path.join(results_dir, "vol_profile_std.npy"), vol_profile_std)

    metrics = {
        "n_crises_used": int(n_crises_used),
        "half_window": int(half_window),
        "theta": {
            "baseline_mean": theta_baseline_mean,
            "baseline_std": theta_baseline_std,
            "pre_mean": pre_theta_mean,
            "pre_std": pre_theta_std,
            "t_stat": float(t_theta),
            "p_value": float(p_theta),
            "cohens_d": d_theta,
        },
        "vol": {
            "baseline_mean": vol_baseline_mean,
            "baseline_std": vol_baseline_std,
            "pre_mean": pre_vol_mean,
            "pre_std": pre_vol_std,
            "t_stat": float(t_vol),
            "p_value": float(p_vol),
            "cohens_d": d_vol,
        },
    }

    metrics_path = os.path.join(results_dir, "metrics_theta_crisis.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Résumé texte
    summary_path = os.path.join(results_dir, "summary_theta_crisis.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Preuve 2 PCPI – Θ(t) comme indicateur précurseur de crise\n")
        f.write("========================================================\n\n")
        f.write(f"Nombre de crises utilisées : {n_crises_used}\n")
        f.write(f"Taille de demi-fenêtre : {half_window}\n\n")

        f.write("Tension Θ(t) :\n")
        f.write(f"  Baseline mean = {theta_baseline_mean:.6f}, std = {theta_baseline_std:.6f}\n")
        f.write(f"  Pré-crise mean = {pre_theta_mean:.6f}, std = {pre_theta_std:.6f}\n")
        f.write(f"  t-stat = {t_theta:.4f}, p-value = {p_theta:.3e}\n")
        f.write(f"  Cohen's d = {d_theta:.4f}\n\n")

        f.write("Volatilité V(t) = |Φ_t - Φ_{t-1}| :\n")
        f.write(f"  Baseline mean = {vol_baseline_mean:.6f}, std = {vol_baseline_std:.6f}\n")
        f.write(f"  Pré-crise mean = {pre_vol_mean:.6f}, std = {pre_vol_std:.6f}\n")
        f.write(f"  t-stat = {t_vol:.4f}, p-value = {p_vol:.3e}\n")
        f.write(f"  Cohen's d = {d_vol:.4f}\n")

    print("Analyse terminée.")
    print(f"Résultats enregistrés dans {metrics_path}")
    print(f"Résumé dans {summary_path}")


if __name__ == "__main__":
    main()
