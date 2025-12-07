import os
import json
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def train_test_split_time_series(X, y, train_ratio=0.8):
    n = len(X)
    n_train = int(train_ratio * n)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, X_test, y_train, y_test


def fit_ar_model(series, p):
    """
    Ajuste un modèle AR(p) par régression linéaire simple.
    Retourne (mse, mae).
    """
    X = []
    y = []
    for t in range(p, len(series)):
        X.append(series[t - p:t])
        y.append(series[t])
    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    return mse, mae


def fit_regression(features, target):
    """
    Régression linéaire multi-features pour prédire target.
    Retourne (mse, mae, coefficients).
    """
    X = np.asarray(features)
    y = np.asarray(target)
    X_train, X_test, y_train, y_test = train_test_split_time_series(X, y)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))

    coefs = {
        "intercept": float(model.intercept_),
        "coeffs": model.coef_.tolist(),
    }
    return mse, mae, coefs


def ks_markov_violation_test(X, M, eps=0.01, q_low=0.3, q_high=0.7):
    """
    Test simple de violation de Markov :
    - on fixe x0 = médiane de X
    - on regarde les instants t où |X_t - x0| < eps
    - on sépare ces points selon M_t (bas vs haut)
    - on compare la distribution de X_{t+1} des deux groupes via KS.

    Retourne dict avec statistique KS, p-value, tailles des groupes, moyennes, écarts-types.
    """
    X = np.asarray(X)
    M = np.asarray(M)
    if len(X) != len(M):
        raise ValueError("X et M doivent avoir la même longueur.")

    x0 = np.median(X)
    idx = np.where(np.abs(X - x0) < eps)[0]
    idx = idx[idx < len(X) - 1]  # pour avoir X_{t+1}

    if len(idx) < 100:
        raise RuntimeError("Trop peu de points proches de x0 pour un test significatif.")

    M_subset = M[idx]

    low_thr = np.quantile(M_subset, q_low)
    high_thr = np.quantile(M_subset, q_high)

    idx_low = idx[M_subset <= low_thr]
    idx_high = idx[M_subset >= high_thr]

    X_next_low = X[idx_low + 1]
    X_next_high = X[idx_high + 1]

    ks_stat, p_value = stats.ks_2samp(X_next_low, X_next_high)

    result = {
        "x0": float(x0),
        "eps": float(eps),
        "q_low": float(q_low),
        "q_high": float(q_high),
        "n_low": int(len(idx_low)),
        "n_high": int(len(idx_high)),
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
        "mean_next_low": float(np.mean(X_next_low)),
        "mean_next_high": float(np.mean(X_next_high)),
        "std_next_low": float(np.std(X_next_low)),
        "std_next_high": float(np.std(X_next_high)),
    }
    return result


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    data_path = os.path.join(data_dir, "pcpi_simulation.npz")
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Fichier de simulation introuvable : {data_path}\n"
            "Veuillez d'abord exécuter simulate_pcpi_system.py"
        )

    data = np.load(data_path)
    phi = data["phi"]
    M = data["M"]
    A = data["A"]
    theta = data["theta"]

    X = phi  # observable publique

    summary = {}

    # 1. AR(p) sur X_t
    print("=== Analyse AR(p) sur X_t ===")
    ar_orders = [1, 2, 3, 5, 10]
    ar_results = {}
    for p in ar_orders:
        mse, mae = fit_ar_model(X, p)
        ar_results[p] = {"mse": mse, "mae": mae}
        print(f"AR({p}): MSE={mse:.6f}, MAE={mae:.6f}")
    summary["ar_models"] = ar_results

    # 2. Régression markovienne X_{t+1} ~ X_t vs PCPI-réduite X_{t+1} ~ X_t + M_t
    print("\n=== Régression markovienne vs PCPI-réduite ===")
    # Construire (X_t, M_t) et X_{t+1}
    X_t = X[:-1]
    M_t = M[:-1]
    y_next = X[1:]

    # Modèle A : X_{t+1} ~ X_t
    mse_A, mae_A, coefs_A = fit_regression(X_t.reshape(-1, 1), y_next)
    print(f"Modèle A (X_(t+1) ~ X_t) : MSE={mse_A:.6f}, MAE={mae_A:.6f}")
    print(f"  Coeffs : {coefs_A}")

    # Modèle B : X_{t+1} ~ X_t + M_t
    feats_B = np.column_stack([X_t, M_t])
    mse_B, mae_B, coefs_B = fit_regression(feats_B, y_next)
    print(f"Modèle B (X_(t+1) ~ X_t + M_t) : MSE={mse_B:.6f}, MAE={mae_B:.6f}")
    print(f"  Coeffs : {coefs_B}")

    summary["regression_markov_vs_pcpi"] = {
        "model_A": {"mse": mse_A, "mae": mae_A, "coeffs": coefs_A},
        "model_B": {"mse": mse_B, "mae": mae_B, "coeffs": coefs_B},
        "relative_improvement_mse": float((mse_A - mse_B) / mse_A),
        "relative_improvement_mae": float((mae_A - mae_B) / mae_A),
    }

    # 3. Test KS de violation de Markov en fonction de M_t
    print("\n=== Test KS de violation de Markov (M_t) ===")
    ks_res = ks_markov_violation_test(X, M, eps=0.01, q_low=0.3, q_high=0.7)
    print("Résultat test KS :")
    for k, v in ks_res.items():
        print(f"  {k}: {v}")
    summary["ks_markov_violation"] = ks_res

    # Sauvegarde des résultats
    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Résumé texte
    summary_txt = os.path.join(results_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Résumé des analyses (preuve 1 PCPI vs Markov)\n")
        f.write("=============================================\n\n")
        f.write("AR(p) sur X_t:\n")
        for p in ar_orders:
            f.write(
                f"  AR({p}): MSE={ar_results[p]['mse']:.6f}, "
                f"MAE={ar_results[p]['mae']:.6f}\n"
            )
        f.write("\nRégression markovienne vs PCPI-réduite:\n")
        f.write(
            f"  Modèle A (X_(t+1) ~ X_t):   MSE={mse_A:.6f}, MAE={mae_A:.6f}\n"
        )
        f.write(
            f"  Modèle B (X_(t+1) ~ X_t + M_t): MSE={mse_B:.6f}, MAE={mae_B:.6f}\n"
        )
        f.write(
            f"  Amélioration relative MSE: {summary['regression_markov_vs_pcpi']['relative_improvement_mse']:.4f}\n"
        )
        f.write(
            f"  Amélioration relative MAE: {summary['regression_markov_vs_pcpi']['relative_improvement_mae']:.4f}\n"
        )
        f.write("\nTest KS de violation de Markov (conditionnel en M_t):\n")
        for k, v in ks_res.items():
            f.write(f"  {k}: {v}\n")

    print("\nRésultats sauvegardés dans :")
    print(f"  {metrics_path}")
    print(f"  {summary_txt}")


if __name__ == '__main__':
    main()
