<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-Matplotlib-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Site_Web-GitHub_Pages-222?logo=github&logoColor=white" />
  <img src="https://img.shields.io/badge/Statut-Complet-2ea44f" />
</p>

<h1 align="center">Métaheuristiques et Algorithmes à Base de Population</h1>

<p align="center">
  <b>Conception, Implémentation et Évaluation Comparative de Métaheuristiques<br>
  pour l'Optimisation sur Espaces Discrets et Continus</b>
</p>

<p align="center">
  Master – Optimisation et Recherche Opérationnelle · ENSET<br>
  <b>Réalisé par :</b> Hicham Ouaouche · <b>Encadré par :</b> Prof. Mestari
</p>

<p align="center">
  <a href="https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/"><img src="https://img.shields.io/badge/🌐_Ouvrir_le_site_interactif-blue?style=for-the-badge" /></a>
</p>

---

## Table des matières

| # | Section |
|---|---|
| 1 | [Description du projet](#1-description-du-projet) |
| 2 | [Structure du projet](#2-structure-du-projet) |
| 3 | [Installation et utilisation](#3-installation-et-utilisation) |
| 4 | [Site web de simulation interactif](#4-site-web-de-simulation-interactif) |
| 5 | [Volet A — Problème binaire](#5-volet-a--problème-binaire) |
| 6 | [Volet B — Algorithme génétique](#6-volet-b--algorithme-génétique) |
| 7 | [Comparaison globale](#7-comparaison-globale) |
| 8 | [Discussion scientifique](#8-discussion-scientifique) |
| 9 | [Perspectives](#9-perspectives) |

---

## 1. Description du projet

Ce mini-projet implémente, analyse et compare **quatre métaheuristiques** appliquées à deux types de problèmes d'optimisation :

| Volet | Problème | Méthodes |
|:---:|---|---|
| **A** | Minimisation sur `{0,1}^10` — espace discret binaire | Descente locale · Recherche taboue · Recuit simulé |
| **B** | Maximisation de `f(x) = sin(x)·exp(sin(x))` sur `[-5, 5]` | Algorithme génétique |

> **Problématique :** *Dans quelle mesure chaque métaheuristique parvient-elle à échapper aux minima locaux, et quels paramètres sont critiques pour la qualité des solutions ?*

---

## 2. Structure du projet

```
RO/
│
├── main.py                           # Point d'entrée — lance toutes les expériences
├── README.md                         # Documentation complète du projet
├── Rapport_Metaheuristiques.docx     # Rapport scientifique (14 pages)
│
├── algorithmes/                      # Package Python
│   ├── __init__.py
│   ├── problem.py                    # Problème binaire : f(s) = Σαᵢbᵢ + Σβᵢⱼbᵢbⱼ
│   ├── local_search.py               # A.1 – Steepest descent
│   ├── tabu_search.py                # A.2 – Recherche taboue (2 stratégies + aspiration)
│   ├── simulated_annealing.py        # A.3 – Recuit simulé (6 configs T₀, λ)
│   └── genetic_algorithm.py          # B   – AG + schèmes de Holland
│
├── figures/                          # 10 figures PNG générées par main.py
│   ├── A0_distribution_couts.png     #   Distribution des 1024 coûts
│   ├── A1_descente_locale.png        #   20 courbes de descente locale
│   ├── A2_tabu_k_strategie.png       #   Effet de k sur la recherche taboue
│   ├── A2_tabu_evolution.png         #   Évolution du coût (taboue)
│   ├── A3_recuit_comparaison.png     #   Comparaison des configs recuit
│   ├── A3_recuit_evolution.png       #   Température + coût (recuit)
│   ├── B0_fonction.png               #   f(x) = sin(x)·exp(sin(x))
│   ├── B2_ag_convergence.png         #   Convergence de l'AG
│   ├── B2_ag_mutation.png            #   Effet du taux de mutation
│   └── C_comparaison_globale.png     #   Comparaison finale
│
├── docs/                             # 🌐 Site web interactif (GitHub Pages)
│   ├── index.html                    #   Accueil — menu des 4 simulations
│   ├── local_search.html             #   Simulation : descente locale
│   ├── tabu_search.html              #   Simulation : recherche taboue
│   ├── simulated_annealing.html      #   Simulation : recuit simulé
│   ├── genetic_algorithm.html        #   Simulation : algorithme génétique
│   └── .nojekyll
│
└── Animé pas à pas/                  # 🎬 5 vidéos de démonstration (MP4)
    ├── index.mp4                     #   Aperçu du site web
    ├── local_search.mp4              #   Démo : descente locale
    ├── tabu_search.mp4               #   Démo : recherche taboue
    ├── simulated_annealing.mp4       #   Démo : recuit simulé
    └── genetic_algorithm.mp4         #   Démo : algorithme génétique
```

---

## 3. Installation et utilisation

### Prérequis

```bash
pip install numpy matplotlib
```

### Lancer toutes les expériences

```bash
python main.py
```

> Le script exécute tous les volets, affiche les résultats dans la console et sauvegarde les figures dans `figures/`.  
> Toutes les expériences sont **reproductibles** (`seed=42`).

### Utilisation individuelle

```python
from algorithmes.problem import BinaryProblem
from algorithmes.local_search import run_local_search, print_local_search_report

prob = BinaryProblem()
_, best_cost, _ = prob.brute_force()
results, stats = run_local_search(prob, n_starts=20, seed=42)
print_local_search_report(results, stats, best_cost)
```

---

## 4. Site web de simulation interactif

<p align="center">
  <a href="https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/">
    <img src="https://img.shields.io/badge/🌐_Accueil-1e293b?style=for-the-badge" />
  </a>
  <a href="https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/local_search.html">
    <img src="https://img.shields.io/badge/📉_Descente_locale-1e293b?style=for-the-badge" />
  </a>
  <a href="https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/tabu_search.html">
    <img src="https://img.shields.io/badge/🚫_Recherche_taboue-1e293b?style=for-the-badge" />
  </a>
  <a href="https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/simulated_annealing.html">
    <img src="https://img.shields.io/badge/🌡️_Recuit_simulé-1e293b?style=for-the-badge" />
  </a>
  <a href="https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/genetic_algorithm.html">
    <img src="https://img.shields.io/badge/🧬_Algorithme_génétique-1e293b?style=for-the-badge" />
  </a>
</p>

Le site propose **4 simulations interactives** (HTML/CSS/JS pur, sans dépendance) hébergées via GitHub Pages :

| Fonctionnalité | Description |
|---|---|
| **Pseudocode interactif** | Ligne active mise en surbrillance en temps réel |
| **Statistiques dynamiques** | Coût courant, meilleur coût, itération, phase |
| **Messages explicatifs** | Interprétation de chaque décision algorithmique |
| **Contrôles** | ▶ Jouer · ⏭ Étape · 🔄 Réinitialiser · Vitesse variable |

> 🎬 **Des vidéos de démonstration** de chaque simulation sont disponibles dans le dossier [`Animé pas à pas/`](Animé%20pas%20à%20pas/).

---

## 5. Volet A — Problème binaire

### 5.1 Modélisation

**Espace :** `s = (b₁, …, b₁₀) ∈ {0,1}¹⁰` · **Voisinage :** Hamming-1

$$f(s) = \sum_{i=1}^{10} \alpha_i b_i + \sum_{1 \le i < j \le 10} \beta_{ij} b_i b_j$$

<details>
<summary><b>Coefficients α et β</b> (cliquer pour ouvrir)</summary>

**Coefficients linéaires α :**

| b₀ | b₁ | b₂ | b₃ | b₄ | b₅ | b₆ | b₇ | b₈ | b₉ |
|---|---|---|---|---|---|---|---|---|---|
| -3.0 | +2.0 | -1.5 | +4.0 | -2.5 | +1.0 | -3.5 | +2.5 | -1.0 | +3.0 |

**Interactions β principales :**

| Paire | βᵢⱼ | Rôle |
|---|---|---|
| (0,1) | +5.0 | Conflit fort |
| (0,4) | -4.0 | Synergie |
| (2,6) | -3.5 | Synergie |
| (3,7) | +4.5 | Conflit |
| (6,9) | -4.0 | Synergie |

</details>

**Vérification par force brute** (2¹⁰ = 1024 solutions) :

```
Minimum global : s* = 1010101011   f* = -19.5000
Minimum local  : s  = 1010111000   f  = -19.0000
```

![Distribution des coûts](figures/A0_distribution_couts.png)

---

### 5.2 Descente locale (A.1)

> Steepest descent — évalue tous les voisins Hamming-1, choisit le meilleur, s'arrête au minimum local.

**20 départs aléatoires ·** `seed=42`

| Métrique | Valeur |
|---|---|
| P(atteindre le global) | **65 %** (13/20) |
| Coût moyen final | -19.3250 |
| Écart-type | 0.2385 |
| Minima atteints | `1010101011` (13×) · `1010111000` (7×) |

![Descente locale](figures/A1_descente_locale.png)

---

### 5.3 Recherche taboue (A.2)

> Deux stratégies : stockage des **solutions** vs stockage des **mouvements inverses** · Critère d'aspiration intégré

**k ∈ {1, 3, 5, 10} · max_iter = 300 · 20 départs**

| k | Stratégie | Coût moy. | P(global) | Dépl. moy. |
|---|---|---|---|---|
| 1 | Solutions | -19.33 | 65% | 300 |
| 1 | **Mouvements** | **-19.50** | **100%** | 300 |
| 3 | Solutions | -19.50 | 100% | 300 |
| 5 | Solutions | -19.50 | 100% | 300 |
| 10 | Mouvements | -19.50 | 100% | ⚠️ 11 |

> k=10 (Mouvements) : les 10 bits deviennent simultanément tabous → blocage.

| | |
|---|---|
| ![Tabu k](figures/A2_tabu_k_strategie.png) | ![Tabu évol](figures/A2_tabu_evolution.png) |

---

### 5.4 Recuit simulé (A.3)

$$p = \begin{cases} 1 & \text{si } \Delta E \le 0 \\ e^{-\Delta E / T} & \text{si } \Delta E > 0 \end{cases} \qquad T_{k+1} = \lambda \cdot T_k$$

> **T₀ estimée automatiquement :** percentile 95 de ΔE sur 500 échantillons, taux d'acceptation cible 80% → **T₀ ≈ 42.57**

**6 configurations · 20 départs · max_iter = 3000**

| Configuration | Coût moy. | P(global) | Itér. moy. |
|---|---|---|---|
| T₀/2, λ=0.90 | -19.38 | 75% | 162 |
| T₀, λ=0.90 | -19.30 | 60% | 168 |
| 2·T₀, λ=0.90 | -19.33 | 65% | 175 |
| T₀, λ=0.85 | -19.23 | 45% | 110 |
| T₀, λ=0.95 | -19.35 | 70% | 344 |
| **T₀, λ=0.99** | **-19.50** | **100%** | **1749** |

| | |
|---|---|
| ![Recuit comp](figures/A3_recuit_comparaison.png) | ![Recuit évol](figures/A3_recuit_evolution.png) |

---

## 6. Volet B — Algorithme génétique

$$f(x) = \sin(x) \cdot e^{\sin(x)}, \quad x \in [-5,\, 5]$$

**Maximum global :** $f^* = e \approx 2.7183$ en $x \approx \pi/2$

![Fonction](figures/B0_fonction.png)

### 6.1 Codage binaire (B.1)

| Paramètre | Valeur |
|---|---|
| Bits par chromosome | 10 |
| Intervalle | [-5, 5] |
| Précision | Δx = 10/1023 ≈ 0.00977 |
| Décodage | $x = -5 + \text{décimal} \times 10 / 1023$ |

### 6.2 Opérateurs génétiques (B.2)

| Composant | Implémentation |
|---|---|
| Sélection | Roulette **ou** Tournoi (k=3) |
| Croisement | Bipoints · probabilité `pc` |
| Mutation | Bit-flip · probabilité `pm` |
| Élitisme | Meilleur individu conservé |

**8 configurations testées :**

| Configuration | f(x) trouvé | Erreur |
|---|---|---|
| pop=20, pm=0.01, roulette | 2.718195 | 8.7×10⁻⁵ |
| pop=50, pm=0.01, roulette | 2.718235 | 4.6×10⁻⁵ |
| **pop=100, pm=0.01, roulette** | **2.718272** | **9.6×10⁻⁶** |
| **pop=50, pm=0.01, tournoi** | **2.718272** | **9.6×10⁻⁶** |
| pop=50, pm=0.05, roulette | 2.718235 | 4.6×10⁻⁵ |
| pop=50, pm=0.10, roulette | 2.718272 | 9.6×10⁻⁶ |

| | |
|---|---|
| ![AG conv](figures/B2_ag_convergence.png) | ![AG mut](figures/B2_ag_mutation.png) |

### 6.3 Analyse des schèmes (B.3)

$$P_{dest}(H) = 1 - \left(1 - p_c \cdot \frac{u(H)}{N-1}\right)\left(1 - p_m\right)^{o(H)}$$

| Schème H | o(H) | u(H) | P(destruction) | Interprétation |
|---|---|---|---|---|
| `1*********` | 1 | 0 | **0.010** | Très robuste |
| `11********` | 2 | 1 | 0.107 | Robuste |
| `1000****1*` | 5 | 8 | **0.725** | Fragile |
| `****1111**` | 4 | 3 | 0.296 | Intermédiaire |

> Confirmation expérimentale du **théorème de Holland** : les schèmes courts et d'ordre faible survivent, les schèmes longs sont systématiquement détruits.

---

## 7. Comparaison globale

### Volet A — Problème binaire (f* = -19.5000)

| Méthode | Coût moy. | P(global) | Itér. moy. |
|---|---|---|---|
| Descente locale | -19.33 | 65% | ~4 |
| Taboue k=5 (Mvt) | **-19.50** | **100%** | 300 |
| Recuit (T₀, λ=0.90) | -19.38 | 75% | 168 |
| Recuit (T₀, λ=0.99) | **-19.50** | **100%** | 1749 |

### Volet B — Algorithme génétique (f* ≈ 2.7183)

| Configuration | Fitness | Erreur |
|---|---|---|
| pop=50, tournoi | 2.718272 | 1.0×10⁻⁵ |
| pop=100, roulette | 2.718272 | 1.0×10⁻⁵ |

![Comparaison globale](figures/C_comparaison_globale.png)

---

## 8. Discussion scientifique

| Méthode | Forces | Limites | Usage recommandé |
|---|---|---|---|
| **Descente locale** | Très rapide, déterministe | Coincée dans les minima locaux (35%) | Borne de référence, composante hybride |
| **Recherche taboue** | Mémoire force la diversification, 100% succès dès k≥3 | k trop grand → blocage | Espaces discrets, critère d'aspiration calibré |
| **Recuit simulé** | Robuste à l'initialisation | λ=0.99 : ×10 en temps de calcul | Surface d'énergie inconnue a priori |
| **Algorithme génétique** | Exploration parallèle, espace continu natif | Convergence prématurée si pop trop petite | Fonctions multimodales continues |

> **Aucune méthode n'est universellement supérieure** — le choix dépend de la nature du problème, de la dimension et des contraintes de temps.

---

## 9. Perspectives

1. **Hybridation** — Initialiser l'AG avec des solutions de la recherche taboue
2. **Adaptation dynamique** — Ajuster λ ou k en cours d'exécution
3. **Passage à l'échelle** — Tester sur n=50 ou n=100 bits
4. **Validation statistique** — Tests de Wilcoxon sur 100 exécutions
5. **Baseline aléatoire** — Random Restart comme référence

---

## Références

- Holland, J.H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Glover, F. (1989). *Tabu Search — Part I*. ORSA Journal on Computing, 1(3), 190–206.
- Kirkpatrick, S., Gelatt, C.D., Vecchi, M.P. (1983). *Optimization by Simulated Annealing*. Science, 220(4598), 671–680.

---

<p align="center">
  <a href="https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/"><b>🌐 Site interactif</b></a> · <a href="https://github.com/hichamouaouche/M-taheuristiques_et_algorithmes_-_base_de_population"><b>📁 Dépôt GitHub</b></a>
  <br><br>
  <sub>Projet réalisé dans le cadre du Master – Optimisation et Recherche Opérationnelle · ENSET</sub>
</p>
