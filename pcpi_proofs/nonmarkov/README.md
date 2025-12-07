# PCPI – Preuve numérique 1 : non-markovianité structurelle

Ce dossier contient le code complet pour reproduire la **première preuve numérique** :
montrer qu’un système gouverné par un champ de cohérence avec mémoire et attracteurs
génère une dynamique **structurellement non-markovienne** lorsqu’on n’observe que la
variable `X_t = Φ_t`.

## 1. Description du système jouet

On simule un système discret en temps avec :

- variable observée : `X_t = Φ_t`,
- variables internes : `M_t` (mémoire), `A_t` (attracteur), `Θ_t = |Φ_t - A_t|` (tension).

Équations (temps discret, t = 0,…,T-1) :

- Potentiel stable non linéaire :  
  V(Φ) = ½ m² Φ² + ¼ λ Φ⁴  
  dV/dΦ = m² Φ + λ Φ³

- Dynamique :

  Φ_{t+1} = Φ_t - α dV/dΦ(Φ_t) + β M_t + γ (A_t - Φ_t) + σ_η η_t  
  M_{t+1} = (1 - λ_M) M_t + λ_M Φ_t  
  A_{t+1} = A_t + ρ (Φ_t - A_t) + σ_ξ ξ_t  
  Θ_t     = |Φ_t - A_t|

où η_t, ξ_t ~ N(0, 1).

On choisit des paramètres raisonnables (fixés dans le script) garantissant une dynamique
non explosive et riche. On simule T = 120 000 pas, puis on jette un burn-in pour ne
garder que la trajectoire stationnaire.

## 2. Scripts fournis

### `simulate_pcpi_system.py`

- Simule le système jouet PCPI.
- Sauvegarde les séries (`phi`, `M`, `A`, `theta`) dans `data/pcpi_simulation.npz`.
- Sauvegarde également `X = phi` dans `data/pcpi_X.npy` pour accès direct.

Usage (dans ce dossier) :

```bash
python simulate_pcpi_system.py
```

Vous pouvez modifier les paramètres (T, burn-in, seed) en ligne de commande.

### `analysis_markov_vs_pcpi.py`

Effectue les analyses suivantes :

1. **AR(p) sur X_t** :  
   - Ajuste des modèles AR(1), AR(2), AR(3), AR(5), AR(10) par régression linéaire.  
   - Calcule MSE et MAE sur un jeu de test (split temporel train/test).  
   - Montre que l’erreur baisse quand p augmente → la dynamique n’est pas Markov d’ordre 1.

2. **Régression markovienne vs PCPI-réduite** :  
   - Modèle A : X_{t+1} ~ X_t  
   - Modèle B : X_{t+1} ~ X_t + M_t  
   - Compare MSE et MAE sur le test.  
   - Montre que l’inclusion de M_t améliore significativement la prédiction.

3. **Test direct de non-markovianité (KS)** :  
   - Fixe une valeur typique x0 (par ex. médiane de X_t).  
   - Sélectionne les temps où |X_t - x0| < ε.  
   - Parmi ces points, sépare les cas "M_t bas" vs "M_t haut" (quantiles).  
   - Compare les distributions de X_{t+1} pour ces deux groupes via un test de Kolmogorov–Smirnov.  
   - Si les distributions diffèrent de façon significative, la loi de X_{t+1} dépend de M_t
     même à X_t fixé → violation de la propriété de Markov.

Les résultats principaux sont imprimés dans la console et résumés dans `results/metrics.json`
et `results/summary.txt`.

Usage :

```bash
python analysis_markov_vs_pcpi.py
```

Assurez-vous d’avoir d’abord exécuté `simulate_pcpi_system.py` pour générer les données.

## 3. Dépendances

- Python 3.x
- numpy
- scipy
- scikit-learn

Vous pouvez installer les dépendances avec :

```bash
pip install -r requirements.txt
```

## 4. Logique scientifique

Cette preuve numérique montre que :

1. Le futur X_{t+1} contient une information sur le passé plus profonde que ce qui est
   encodé dans X_t seul.
2. La variable de mémoire interne M_t améliore significativement la prédiction de X_{t+1}.
3. Pour un même X_t (dans un petit voisinage), la distribution de X_{t+1} dépend
   fortement de la valeur de M_t, ce qui viole directement la propriété de Markov
   en X_t.

Cela constitue une démonstration numérique claire que les dynamiques issues d’un champ
de cohérence avec mémoire et attracteurs ne sont pas compatibles avec un modèle markovien
simple sur la variable observée X_t.

## 5. Organisation du dossier

- `simulate_pcpi_system.py` : simulation du système jouet PCPI
- `analysis_markov_vs_pcpi.py` : analyses AR, régression, test KS
- `data/` : fichiers de simulation (NPZ/NPY)
- `results/` : métriques et résumés des analyses
- `requirements.txt` : dépendances Python minimales

Ce kit est autonome : il suffit de cloner/copier ce dossier, installer les
dépendances, lancer la simulation puis l’analyse pour reproduire la preuve numérique.
