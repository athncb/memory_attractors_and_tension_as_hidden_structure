# PCPI – Preuve numérique 3 (version corrigée) : variables de cohérence sur données de marché réelles

Cette version corrige la dépendance à la colonne `Adj Close` dans les données Yahoo Finance.
Désormais, le script :

- détecte automatiquement si les colonnes sont simples ou en MultiIndex,
- utilise `Adj Close` si disponible,
- sinon se rabat sur `Close`.

Le reste de la logique reste identique : construction de `phi`, `A`, `theta`, `M`, `ret`, `vol`,
puis analyse PCPI vs baseline sur la prédiction d'une forte baisse à J+1.
