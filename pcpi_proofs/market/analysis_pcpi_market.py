import os
import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss


def time_series_train_test_split(X, y, train_ratio=0.7):
    n = len(y)
    n_train = int(train_ratio * n)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    return X_train, X_test, y_train, y_test


def fit_logistic_and_eval(X, y, train_ratio=0.7):
    """
    Ajuste une régression logistique simple et retourne :
    - AUC ROC
    - Brier score
    - Log-loss
    """
    X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, train_ratio=train_ratio)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    proba_test = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, proba_test)
    brier = brier_score_loss(y_test, proba_test)
    ll = log_loss(y_test, proba_test)

    return {
        "auc": float(auc),
        "brier": float(brier),
        "log_loss": float(ll),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyse PCPI vs baseline sur données de marché réelles.")
    parser.add_argument("--ticker", type=str, default="^GSPC", help="Même ticker que pour la préparation (ex: ^GSPC)")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Fraction de données pour l'entraînement")
    args = parser.parse_args()

    ticker = args.ticker
    train_ratio = args.train_ratio

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    suffix = ticker.replace("^", "").replace("-", "_")
    npz_path = os.path.join(data_dir, f"pcpi_market_features_{suffix}.npz")
    if not os.path.exists(npz_path):
        raise FileNotFoundError(
            f"Fichier {npz_path} introuvable. "
            "Veuillez d'abord exécuter download_and_prepare_market_data.py avec le même ticker."
        )

    data = np.load(npz_path)
    phi = data["phi"]
    A = data["A"]
    theta = data["theta"]
    M = data["M"]
    ret = data["ret"]
    vol = data["vol"]
    event_next = data["event_next"]

    # shapes cohérentes
    n = len(event_next)
    phi = phi[:n]
    A = A[:n]
    theta = theta[:n]
    M = M[:n]
    ret = ret[:n]
    vol = vol[:n]

    # Modèle baseline : event_{t+1} ~ ret_t + vol_t
    X_base = np.column_stack([ret, vol])
    y = event_next
    mask = np.isfinite(X_base).all(axis=1) & np.isfinite(y)
    X_base = X_base[mask]
    y = y[mask]

    print(f"Nombre de points utilisables après filtrage : {len(y)}")

    metrics_base = fit_logistic_and_eval(X_base, y, train_ratio=train_ratio)

    # Modèle enrichi PCPI : event_{t+1} ~ ret_t + vol_t + theta_t + M_t
    X_pcpi = np.column_stack([ret, vol, theta, M])
    X_pcpi = X_pcpi[mask]
    metrics_pcpi = fit_logistic_and_eval(X_pcpi, y, train_ratio=train_ratio)

    metrics = {
        "ticker": ticker,
        "train_ratio": train_ratio,
        "baseline": metrics_base,
        "pcpi_enriched": metrics_pcpi,
        "relative_improvement": {
            "auc": float(metrics_pcpi["auc"] - metrics_base["auc"]),
            "brier": float(metrics_base["brier"] - metrics_pcpi["brier"]),
            "log_loss": float(metrics_base["log_loss"] - metrics_pcpi["log_loss"]),
        },
    }

    suffix = ticker.replace("^", "").replace("-", "_")
    metrics_path = os.path.join(results_dir, f"metrics_pcpi_market_{suffix}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(results_dir, f"summary_pcpi_market_{suffix}.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Preuve 3 PCPI – données de marché pour {ticker}\n")
        f.write("==============================================\n\n")
        f.write("Modèle baseline (event_{t+1} ~ ret_t + vol_t) :\n")
        f.write(f"  AUC       : {metrics_base['auc']:.4f}\n")
        f.write(f"  Brier     : {metrics_base['brier']:.6f}\n")
        f.write(f"  Log-loss  : {metrics_base['log_loss']:.6f}\n")
        f.write(f"  n_train   : {metrics_base['n_train']}\n")
        f.write(f"  n_test    : {metrics_base['n_test']}\n\n")

        f.write("Modèle enrichi PCPI (event_{t+1} ~ ret_t + vol_t + theta_t + M_t) :\n")
        f.write(f"  AUC       : {metrics_pcpi['auc']:.4f}\n")
        f.write(f"  Brier     : {metrics_pcpi['brier']:.6f}\n")
        f.write(f"  Log-loss  : {metrics_pcpi['log_loss']:.6f}\n")
        f.write(f"  n_train   : {metrics_pcpi['n_train']}\n")
        f.write(f"  n_test    : {metrics_pcpi['n_test']}\n\n")

        f.write("Améliorations (PCPI - baseline) :\n")
        f.write(f"  Δ AUC            : {metrics['relative_improvement']['auc']:.4f}\n")
        f.write(f"  Δ Brier (gain)   : {metrics['relative_improvement']['brier']:.6f}\n")
        f.write(f"  Δ Log-loss (gain): {metrics['relative_improvement']['log_loss']:.6f}\n")

    print("Analyse terminée.")
    print(f"Métriques sauvegardées dans : {metrics_path}")
    print(f"Résumé dans : {summary_path}")


if __name__ == "__main__":
    main()
