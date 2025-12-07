# PCPI – Preuve numérique 2 : Θ(t) comme indicateur précurseur de crise

Ce dossier contient le code complet pour reproduire la **deuxième preuve numérique** :
montrer que, dans un système cohérent avec attracteurs et stress, la **tension Θ(t)**
monte systématiquement **avant** les bifurcations de régime, bien plus clairement
qu’un indicateur naïf de volatilité.

## 1. Description du système jouet

On considère un système discret en temps avec :

- état observable :    Φ_t (on peut aussi le noter X_t),
- attracteur lent :    A_t,
- tension :            Θ_t = |Φ_t - A_t|,
- stress accumulé :    S_t,
- régime :             r_t ∈ {−1, +1}.

### 1.1. Dynamique

Pour t = 0,…,T−1 :

- Attracteur structurel du régime :  A*_t = r_t · A_star
- Mise à jour de l’attracteur lent :

  A_{t+1} = A_t + γ (A*_t − A_t)

- Mise à jour de l’état :

  Φ_{t+1} = Φ_t + ε (A_t − Φ_t) + σ η_t

  où η_t ~ N(0, 1).

- Tension instantanée :

  Θ_t = |Φ_t − A_t|

- Stress (mémoire exponentielle de la tension) :

  S_{t+1} = λ_s S_t + (1 − λ_s) Θ_t

### 1.2. Règle de crise / changement de régime

On fixe un seuil de stress S_c. Lorsque S_{t+1} dépasse ce seuil, on déclenche
une **crise** :

- r_{t+1} = − r_t
- S_{t+1} = 0
- Φ_{t+1} reçoit un léger choc additionnel.

La série r_t alterne donc entre deux régimes, et chaque basculement correspond
à une « crise ».

## 2. Scripts fournis

### `simulate_pcpi_system_proof2.py`

- Simule le système décrit ci-dessus.
- Sauvegarde les séries dans `data/pcpi_proof2_simulation.npz`.

Usage :

```bash
python simulate_pcpi_system_proof2.py
```

### `analysis_theta_crisis.py`

Effectue les analyses suivantes :

1. Détection des instants de crise (changement de signe de r_t).
2. Construction de fenêtres autour de chaque crise : [t_c − W, …, t_c + W].
3. Calcul du **profil moyen de Θ(Δt)**, où Δt = t − t_c.
4. Calcul d’une baseline « loin des crises ».
5. Comparaison de la tension moyenne pré-crise (par ex. Δt ∈ [−W,−1]) à la baseline :
   - moyenne, écart-type,
   - test t de Student,
   - taille d’effet (Cohen’s d).
6. Comparaison avec un indicateur de « volatilité » naïf :

   V_t = |Φ_t − Φ_{t−1}|

   pour montrer que V_t reste quasi plat avant la crise et ne fait que réagir
   au moment du choc.

Les résultats sont enregistrés dans :

- `results/theta_profile.npy`         : profil moyen de Θ(Δt),
- `results/vol_profile.npy`           : profil moyen de V(Δt),
- `results/metrics_theta_crisis.json` : statistiques détaillées,
- `results/summary_theta_crisis.txt`  : résumé humain.

Usage :

```bash
python analysis_theta_crisis.py
```

## 3. Dépendances

- Python 3.x
- numpy
- scipy

Installation :

```bash
pip install -r requirements.txt
```

## 4. Logique scientifique

Cette preuve numérique montre que :

1. La tension de cohérence Θ(t) augmente de manière significative **avant** les crises.
2. La moyenne de Θ(t) sur une fenêtre pré-crise [−W,−1] est nettement supérieure
   à son niveau de base loin des crises, avec une taille d’effet élevée.
3. Un indicateur naïf de volatilité V(t) ne montre pas cette montée précoce :
   il reste proche de son niveau moyen jusqu’au choc lui-même.

Cela illustre le rôle de Θ(t) comme **indicateur précurseur de bifurcation de régime**,
conformément à l’intuition PCPI : la crise est précédée par une montée de tension entre
l’état courant et l’attracteur, montée qui n’est pas visible dans les fluctuations
instantanées de Φ_t seules.
