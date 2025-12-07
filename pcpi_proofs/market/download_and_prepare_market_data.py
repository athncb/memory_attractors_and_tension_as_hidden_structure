import os
import argparse
import numpy as np
import pandas as pd
import yfinance as yf


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _extract_close_series(df: pd.DataFrame) -> pd.Series:
    """
    Récupère une série de prix de clôture à partir du DataFrame Yahoo Finance,
    en gérant :
      - colonnes simples,
      - colonnes MultiIndex,
      - présence ou non de 'Adj Close'.
    """
    cols = df.columns

    # Cas 1 : colonnes simples (Index)
    if not isinstance(cols, pd.MultiIndex):
        for name in ["Adj Close", "Close", "close", "adjclose"]:
            if name in cols:
                return df[name].dropna()
        raise KeyError(f"Aucune colonne de type Close/Adj Close trouvée dans les colonnes simples : {list(cols)}")

    # Cas 2 : colonnes MultiIndex
    # Ton cas : niveau 0 = type de prix, niveau 1 = ticker
    level0 = cols.get_level_values(0)
    for candidate in ["Adj Close", "Close", "close", "adjclose"]:
        if candidate in level0:
            sub = df[candidate]  # sélectionne toutes les colonnes dont le niveau 0 = candidate
            # Si plusieurs tickers : DataFrame, on prend la première colonne
            if isinstance(sub, pd.DataFrame):
                return sub.iloc[:, 0].dropna()
            return sub.dropna()

    # Fallback : on tente quand même sur chaque niveau (cas exotiques)
    for level in range(cols.nlevels):
        level_vals = cols.get_level_values(level)
        for candidate in ["Adj Close", "Close", "close", "adjclose"]:
            if candidate in level_vals:
                sub = df.xs(candidate, axis=1, level=level)
                if isinstance(sub, pd.DataFrame):
                    return sub.iloc[:, 0].dropna()
                return sub.dropna()

    raise KeyError(f"Aucune colonne de type Close/Adj Close trouvée dans les colonnes MultiIndex : {list(cols)}")


def prepare_features(
    df: pd.DataFrame,
    ema_span_attractor: int = 50,
    ema_span_memory: int = 20,
    vol_window: int = 20,
    q_neg: float = 0.05,
):
    """
    À partir d'un DataFrame de prix, construit :
    - phi_t   : log-prix
    - A_t     : EMA(phi_t)
    - theta_t : |phi_t - A_t|
    - M_t     : EMA(theta_t)
    - ret_t   : rendement journalier (log)
    - vol_t   : volatilité locale (rolling std des rendements)
    - event_{t+1} : indicatrice de forte baisse (ret_{t+1} < quantile q_neg)
    """
    close = _extract_close_series(df)
    phi = np.log(close)

    A = compute_ema(phi, span=ema_span_attractor)
    theta = (phi - A).abs()
    M = compute_ema(theta, span=ema_span_memory)

    ret = phi.diff()
    vol = ret.rolling(window=vol_window).std()

    # cible : événement de forte baisse à horizon 1 jour
    ret_next = ret.shift(-1)
    threshold = ret_next.quantile(q_neg)
    event = (ret_next < threshold).astype(int)

    features = pd.DataFrame(
        {
            "phi": phi,
            "A": A,
            "theta": theta,
            "M": M,
            "ret": ret,
            "vol": vol,
            "event_next": event,
        }
    ).dropna()

    return features, threshold


def main():
    parser = argparse.ArgumentParser(description="Télécharger et préparer des données de marché pour la preuve 3 PCPI.")
    parser.add_argument("--ticker", type=str, default="^GSPC", help="Ticker Yahoo Finance (ex: ^GSPC, SPY, BTC-USD)")
    parser.add_argument("--start", type=str, default="1990-01-01", help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Date de fin (YYYY-MM-DD, défaut = aujourd'hui)")
    parser.add_argument("--q_neg", type=float, default=0.05, help="Quantile pour définir une forte baisse (ex: 0.05)")
    args = parser.parse_args()

    ticker = args.ticker
    start = args.start
    end = args.end
    q_neg = args.q_neg

    print(f"Téléchargement des données pour {ticker} depuis Yahoo Finance...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise RuntimeError(f"Aucune donnée téléchargée pour le ticker {ticker}.")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    features, threshold = prepare_features(data, q_neg=q_neg)

    out_suffix = ticker.replace("^", "").replace("-", "_")
    out_path_npz = os.path.join(data_dir, f"pcpi_market_features_{out_suffix}.npz")
    np.savez_compressed(
        out_path_npz,
        phi=features["phi"].values,
        A=features["A"].values,
        theta=features["theta"].values,
        M=features["M"].values,
        ret=features["ret"].values,
        vol=features["vol"].values,
        event_next=features["event_next"].values.astype(int),
    )

    out_path_csv = os.path.join(data_dir, f"pcpi_market_features_{out_suffix}.csv")
    features.to_csv(out_path_csv, index=True)

    print(f"Features sauvegardés dans : {out_path_npz}")
    print(f"Copie lisible dans : {out_path_csv}")
    print(f"Seuil de forte baisse (quantile {q_neg:.2f}) sur ret_next : {threshold:.6f}")


if __name__ == "__main__":
    main()
