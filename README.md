<div align="center">

# 🧬 Métaheuristiques & Algorithmes à Base de Population

**Conception, Implémentation et Évaluation Comparative de Métaheuristiques**  
**pour l'Optimisation sur Espaces Discrets et Continus**

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?style=flat-square&logo=numpy)](https://numpy.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557c?style=flat-square)](https://matplotlib.org)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Live-2ea44f?style=flat-square&logo=github)](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/)
[![Status](https://img.shields.io/badge/Statut-Complet%20✓-success?style=flat-square)](.)
[![License](https://img.shields.io/badge/Licence-Académique-blue?style=flat-square)](.)

---

🎓 **Filière :** Master – Optimisation et Recherche Opérationnelle  
🏛️ **Établissement :** ENSET  
👨‍💻 **Réalisé par :** Hicham Ouaouche  
👨‍🏫 **Encadré par :** Prof. Mestari

---

### 🌐 [Ouvrir le Site de Simulation Interactif](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/)

[![Descente locale](https://img.shields.io/badge/📉_Descente_locale-Simulation-3b82f6?style=for-the-badge)](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/local_search.html)
[![Recherche taboue](https://img.shields.io/badge/🚫_Recherche_taboue-Simulation-8b5cf6?style=for-the-badge)](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/tabu_search.html)
[![Recuit simulé](https://img.shields.io/badge/🌡️_Recuit_simulé-Simulation-f59e0b?style=for-the-badge)](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/simulated_annealing.html)
[![Algorithme génétique](https://img.shields.io/badge/🧬_Algo_Génétique-Simulation-10b981?style=for-the-badge)](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/genetic_algorithm.html)

</div>

---

## 📋 Table des matières

- [🎯 Objectifs et problématique](#-objectifs-et-problématique)
- [📁 Structure du projet](#-structure-du-projet)
- [⚙️ Installation et utilisation](#️-installation-et-utilisation)
- [🌐 Site web de simulation interactif](#-site-web-de-simulation-interactif)
- [🎬 Démonstrations vidéo](#-démonstrations-vidéo)
- [🔷 Volet A — Problème binaire](#-volet-a--problème-binaire)
  - [Modélisation du problème](#-modélisation-du-problème)
  - [A.1 Descente locale](#a1--descente-locale-steepest-descent)
  - [A.2 Recherche taboue](#a2--recherche-taboue)
  - [A.3 Recuit simulé](#a3--recuit-simulé)
- [🔶 Volet B — Algorithme génétique](#-volet-b--algorithme-génétique)
  - [B.1 Codage binaire](#b1--codage-binaire)
  - [B.2 Opérateurs génétiques](#b2--opérateurs-génétiques)
  - [B.3 Analyse des schèmes](#b3--analyse-des-schèmes-de-holland)
- [📊 Comparaison globale](#-comparaison-globale)
- [🔬 Discussion scientifique](#-discussion-scientifique)
- [🚀 Perspectives](#-perspectives)
- [📚 Références](#-références)

---

## 🎯 Objectifs et problématique

Ce projet implémente, analyse et compare expérimentalement **quatre métaheuristiques** sur deux classes de problèmes d'optimisation :

| | Volet | Problème | Algorithmes |
|:---:|:---:|---|---|
| 🔷 | **A** | Minimisation sur `{0,1}^10` — espace discret binaire | Descente locale, Recherche taboue, Recuit simulé |
| 🔶 | **B** | Maximisation de `f(x) = sin(x)·eˢⁱⁿ⁽ˣ⁾` sur `[-5, 5]` | Algorithme génétique |

**Problématique centrale :**
> *Dans quelle mesure chaque métaheuristique parvient-elle à échapper aux minima locaux ? Quels paramètres sont critiques pour la qualité des solutions ? Comment comparer rigoureusement des approches de natures différentes ?*

**Objectifs pédagogiques :**
1. Modéliser un problème d'optimisation sur espace discret binaire
2. Analyser les limites de la recherche locale gloutonne face aux minima locaux
3. Implémenter et paramétrer une recherche taboue et un recuit simulé
4. Utiliser un algorithme génétique pour l'optimisation d'une fonction non linéaire
5. Comparer rigoureusement plusieurs approches selon des critères quantitatifs
6. Rédiger une analyse critique fondée sur des résultats expérimentaux reproductibles

---

## 📁 Structure du projet

```
RO/
│
├── 📄 main.py                          # Point d'entrée — lance toutes les expériences
├── 📄 README.md                        # Documentation complète du projet
├── 📄 Rapport_Metaheuristiques.docx    # Rapport scientifique complet (14 pages)
│
├── 📦 algorithmes/                     # Package Python des algorithmes
│   ├── __init__.py                     #   Initialisation du package
│   ├── problem.py                      #   Problème binaire : classe BinaryProblem
│   ├── local_search.py                 #   A.1 – Steepest descent (20 départs)
│   ├── tabu_search.py                  #   A.2 – Recherche taboue (2 stratégies + aspiration)
│   ├── simulated_annealing.py          #   A.3 – Recuit simulé (6 configs T₀, λ)
│   └── genetic_algorithm.py           #   B   – AG complet + schèmes de Holland
│
├── 📊 figures/                         # 10 figures PNG générées par main.py
│   ├── A0_distribution_couts.png       #   Distribution des 1024 coûts de l'espace binaire
│   ├── A1_descente_locale.png          #   20 courbes de descente locale
│   ├── A2_tabu_k_strategie.png         #   Effet de la taille k de la liste taboue
│   ├── A2_tabu_evolution.png           #   Évolution du coût — recherche taboue
│   ├── A3_recuit_comparaison.png       #   Comparaison des 6 configs recuit simulé
│   ├── A3_recuit_evolution.png         #   Température + coût — recuit simulé
│   ├── B0_fonction.png                 #   f(x) = sin(x)·exp(sin(x)) sur [-5, 5]
│   ├── B2_ag_convergence.png           #   Convergence fitness — algorithme génétique
│   ├── B2_ag_mutation.png              #   Effet du taux de mutation pm
│   └── C_comparaison_globale.png       #   Synthèse comparative finale
│
├── 🌐 docs/                            # Site web interactif (GitHub Pages)
│   ├── index.html                      #   Accueil — menu des 4 simulations
│   ├── local_search.html               #   Simulation interactive : descente locale
│   ├── tabu_search.html                #   Simulation interactive : recherche taboue
│   ├── simulated_annealing.html        #   Simulation interactive : recuit simulé
│   ├── genetic_algorithm.html          #   Simulation interactive : algorithme génétique
│   └── .nojekyll                       #   GitHub Pages — désactive Jekyll
│
└── 🎬 Animé pas à pas/                 # Vidéos de démonstration (MP4)
    ├── index.mp4                       #   Vue d'ensemble du site web
    ├── local_search.mp4                #   Démo pas à pas : descente locale
    ├── tabu_search.mp4                 #   Démo pas à pas : recherche taboue
    ├── simulated_annealing.mp4         #   Démo pas à pas : recuit simulé
    └── genetic_algorithm.mp4           #   Démo pas à pas : algorithme génétique
```

---

## ⚙️ Installation et utilisation

### Prérequis

```bash
pip install numpy matplotlib
```

### Lancer toutes les expériences

```bash
python main.py
```

> Le script exécute automatiquement tous les volets A et B, affiche les tableaux dans la console, et sauvegarde les 10 figures dans `figures/`.  
> ✅ **Reproductible** — toutes les expériences utilisent `seed=42`.

### Utiliser un module individuellement

```python
from algorithmes.problem import BinaryProblem
from algorithmes.local_search import run_local_search, print_local_search_report

prob = BinaryProblem()
_, best_cost, _ = prob.brute_force()
results, stats = run_local_search(prob, n_starts=20, seed=42)
print_local_search_report(results, stats, best_cost)
```

### Accès sans installation

👉 **[Site de simulation interactif](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/)** — aucun prérequis, directement dans le navigateur.

---

## 🌐 Site web de simulation interactif

Le site est hébergé sur **GitHub Pages** depuis le dossier `docs/` et propose **4 simulations interactives** développées en HTML/CSS/JavaScript pur (aucune dépendance externe).

### Simulations disponibles

| Algorithme | Lien direct | Description |
|---|:---:|---|
| 📉 Descente locale | [Ouvrir →](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/local_search.html) | Steepest descent sur paysage de coût interactif |
| 🚫 Recherche taboue | [Ouvrir →](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/tabu_search.html) | Liste taboue configurable + critère d'aspiration |
| 🌡️ Recuit simulé | [Ouvrir →](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/simulated_annealing.html) | Contrôle de T₀ et λ en temps réel |
| 🧬 Algorithme génétique | [Ouvrir →](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/genetic_algorithm.html) | Évolution de population sur f(x) = sin(x)·eˢⁱⁿ⁽ˣ⁾ |

### Fonctionnalités communes

| Fonctionnalité | Description |
|---|---|
| 📝 **Pseudocode interactif** | Ligne courante mise en surbrillance synchronisée avec l'animation |
| 📈 **Statistiques temps réel** | Coût courant, meilleur coût, numéro d'itération, phase |
| 💬 **Explications dynamiques** | Message contextuel décrivant chaque décision algorithmique |
| ⚡ **Contrôles** | ▶ Jouer · ⏭ Étape · 🔄 Réinitialiser · 🐢🐇 Vitesse variable |
| 🎨 **Visualisation** | Paysage de fitness, trajectoire, population (selon l'algo) |

---

## 🎬 Démonstrations vidéo

Le dossier [`Animé pas à pas/`](Animé%20pas%20à%20pas/) contient **5 vidéos MP4** enregistrées depuis le site, montrant le déroulement étape par étape de chaque algorithme.

| # | Fichier | Contenu |
|:---:|---|---|
| 1 | [`index.mp4`](Animé%20pas%20à%20pas/index.mp4) | Vue d'ensemble du site web et navigation entre les simulations |
| 2 | [`local_search.mp4`](Animé%20pas%20à%20pas/local_search.mp4) | Descente vers un minimum local, chemin parcouru, arrêt steepest descent |
| 3 | [`tabu_search.mp4`](Animé%20pas%20à%20pas/tabu_search.mp4) | Mise à jour liste taboue, critère d'aspiration (violet), évasion minima locaux |
| 4 | [`simulated_annealing.mp4`](Animé%20pas%20à%20pas/simulated_annealing.mp4) | Acceptation probabiliste `exp(-ΔE/T)`, refroidissement progressif |
| 5 | [`genetic_algorithm.mp4`](Animé%20pas%20à%20pas/genetic_algorithm.mp4) | Évolution population, sélection/croisement/mutation, convergence fitness |

---

## 🔷 Volet A — Problème binaire

### 📐 Modélisation du problème

**Espace de recherche :** `s = (b₁, b₂, …, b₁₀) ∈ {0,1}¹⁰` — 2¹⁰ = **1024 solutions**  
**Voisinage :** distance de Hamming-1 → **n = 10 voisins** par solution

**Fonction de coût à minimiser :**

$$f(s) = \sum_{i=1}^{10} \alpha_i \, b_i \;+\; \sum_{1 \leq i < j \leq 10} \beta_{ij} \, b_i \, b_j$$

<details>
<summary><b>📊 Voir les coefficients α et β complets</b></summary>

**Coefficients linéaires α — créent des tensions individuelles :**

| Bit | b₀ | b₁ | b₂ | b₃ | b₄ | b₅ | b₆ | b₇ | b₈ | b₉ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **αᵢ** | -3.0 | +2.0 | -1.5 | +4.0 | -2.5 | +1.0 | -3.5 | +2.5 | -1.0 | +3.0 |
| **Tendance** | → 1 | → 0 | → 1 | → 0 | → 1 | → 0 | → 1 | → 0 | → 1 | → 0 |

**Principales interactions β — créent des conflits locaux :**

| Paire (i,j) | βᵢⱼ | Type | Effet |
|:---:|:---:|:---:|---|
| (0, 1) | +5.0 | ⚡ Conflit fort | Pénalise fortement b₀=b₁=1 |
| (0, 4) | -4.0 | 🤝 Synergie | Récompense b₀=b₄=1 |
| (2, 6) | -3.5 | 🤝 Synergie | Récompense b₂=b₆=1 |
| (3, 7) | +4.5 | ⚡ Conflit | Pénalise b₃=b₇=1 |
| (6, 9) | -4.0 | 🤝 Synergie | Récompense b₆=b₉=1 |

> Ces interactions créent intentionnellement plusieurs **minima locaux distincts** du minimum global — ce qui rend le problème non trivial et pertinent pour tester les métaheuristiques.

</details>

**Résultats de la force brute (vérification exhaustive) :**

```
✅ Minimum global  :  s* = 1010101011   f* = -19.5000
⚠️ Minimum local   :  s  = 1010111000   f  = -19.0000
   Nombre total de minima locaux : voir figure ci-dessous
```

![Distribution des coûts](figures/A0_distribution_couts.png)

---

### A.1 — Descente locale (Steepest Descent)

**Principe :** À chaque itération, évaluer les **n=10 voisins Hamming-1**, choisir le meilleur, s'arrêter dès qu'aucun voisin n'améliore la solution courante (minimum local atteint).

```
ALGORITHME steepest_descent(s₀) :
  s ← s₀ ;  c ← f(s)
  TANT QUE il existe un voisin s' tel que f(s') < f(s) :
    s ← argmin f(N(s))
    c ← f(s)
  RETOURNER s, c
```

**Configuration expérimentale :** 20 initialisations aléatoires · `seed=42`

**Résultats détaillés :**

| # | Solution initiale | Solution finale | Coût final | Global ? |
|:---:|:---:|:---:|:---:|:---:|
| 1 | `0110010100` | `1010111000` | -19.0000 | ❌ |
| 2 | `1111111010` | `1010101011` | **-19.5000** | ✅ |
| 3 | `1001110110` | `1010101011` | **-19.5000** | ✅ |
| 4 | `0001101101` | `1010101011` | **-19.5000** | ✅ |
| 5 | `0110010111` | `1010101011` | **-19.5000** | ✅ |
| 6 | `1000001011` | `1010101011` | **-19.5000** | ✅ |
| 7 | `1101001000` | `1010111000` | -19.0000 | ❌ |
| 8 | `1000100011` | `1010101011` | **-19.5000** | ✅ |
| 9 | `1001110011` | `1010101011` | **-19.5000** | ✅ |
| 10 | `0110100110` | `1010101011` | **-19.5000** | ✅ |
| 11 | `1010111101` | `1010111000` | -19.0000 | ❌ |
| 12 | `0110110000` | `1010111000` | -19.0000 | ❌ |
| 13 | `0110110111` | `1010101011` | **-19.5000** | ✅ |
| 14 | `1101101000` | `1010111000` | -19.0000 | ❌ |
| 15 | `1000110010` | `1010101011` | **-19.5000** | ✅ |
| 16 | `1010010111` | `1010101011` | **-19.5000** | ✅ |
| 17 | `1100010000` | `1010111000` | -19.0000 | ❌ |
| 18 | `0011101011` | `1010101011` | **-19.5000** | ✅ |
| 19 | `1001000000` | `1010111000` | -19.0000 | ❌ |
| 20 | `1110100001` | `1010101011` | **-19.5000** | ✅ |

**Analyse statistique :**

| Métrique | Valeur |
|---|:---:|
| **P(atteindre le minimum global)** | **65 %** (13/20) |
| Coût moyen final | -19.3250 |
| Écart-type des coûts finaux | 0.2385 |
| Minima locaux distincts atteints | 2 |
| `1010101011` atteint | 13 fois |
| `1010111000` atteint | 7 fois |
| Itérations moyennes par départ | ~4 |

> **Conclusion A.1 :** La descente locale est très rapide (O(n) par itération) mais reste bloquée dans un minimum local dans **35% des cas**, démontrant sa forte sensibilité à l'initialisation.

![Descente locale](figures/A1_descente_locale.png)

---

### A.2 — Recherche taboue

**Principe :** Extension de la descente locale avec une **liste taboue** de mémoire à court terme qui interdit de revisiter des solutions/mouvements récents, forçant la diversification.

**Deux stratégies de mémoire implémentées :**

| Stratégie | Mémoire stockée | Complexité mémoire |
|---|---|:---:|
| **Solutions** | k dernières solutions complètes | O(k × n) |
| **Mouvements** | k derniers indices de bits flippés | O(k) |

**Critère d'aspiration :** Un mouvement tabou est accepté si la solution obtenue est **meilleure que le meilleur global** connu.

```
ALGORITHME tabu_search(s₀, k, max_iter) :
  s ← s₀ ;  best ← s₀ ;  L ← {s₀}
  POUR iter = 1 À max_iter :
    s* ← argmin{ f(s') | s' ∈ N(s), s' ∉ L OU f(s') < f(best) }
    s ← s* ;  L.add(s)  // |L| ≤ k
    SI f(s) < f(best) : best ← s
  RETOURNER best
```

**Paramètres :** k ∈ {1, 3, 5, 10} · max_iter = 300 · 20 départs · `seed=42`

**Résultats comparatifs :**

| k | Stratégie | Coût moyen | Coût min | P(global) | Déplacements moy. |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | Solutions | -19.3250 | -19.5000 | 65% | 300 |
| 1 | **Mouvements** | **-19.5000** | **-19.5000** | **100%** | 300 |
| 3 | Solutions | -19.5000 | -19.5000 | 100% | 300 |
| 3 | Mouvements | -19.5000 | -19.5000 | 100% | 300 |
| 5 | Solutions | -19.5000 | -19.5000 | 100% | 300 |
| 5 | Mouvements | -19.5000 | -19.5000 | 100% | 300 |
| 10 | Solutions | -19.5000 | -19.5000 | 100% | 300 |
| 10 | Mouvements | -19.5000 | -19.5000 | 100% | ⚠️ 11 |

> ⚠️ **k=10 Mouvements :** Avec n=10 bits, interdire les 10 derniers flip bloque l'ensemble des mouvements possibles. L'algorithme s'arrête après seulement **11 itérations**.

**Analyse par les figures :**

| Effet de k sur la stratégie | Évolution du coût (k=5, Mouvements) |
|:---:|:---:|
| ![Tabu k](figures/A2_tabu_k_strategie.png) | ![Tabu évol](figures/A2_tabu_evolution.png) |

> **Conclusion A.2 :** La recherche taboue surpasse nettement la descente locale — dès k≥3, les deux stratégies atteignent systématiquement le minimum global (**P=100%**). La stratégie Mouvements est plus économique en mémoire.

---

### A.3 — Recuit simulé

**Principe :** Acceptation probabiliste des mouvements dégradants selon une température décroissante, permettant d'échapper aux minima locaux en début d'exécution.

**Règle d'acceptation de Metropolis :**

$$p(\text{accepter}) = \begin{cases} 1 & \text{si } \Delta E \leq 0 \quad \text{(amélioration)} \\ e^{-\Delta E / T} & \text{si } \Delta E > 0 \quad \text{(dégradation)} \end{cases}$$

**Calendrier de refroidissement géométrique :**

$$T_{k+1} = \lambda \cdot T_k, \quad 0.85 \leq \lambda \leq 0.99$$

**Estimation de T₀ (méthode empirique) :**

```python
# estimate_T0() : percentile 95 de ΔE sur 500 échantillons aléatoires
# Cible : P(accepter la pire dégradation) ≈ 80%
# T₀ = -ΔE_max / ln(0.80)
T₀ ≈ 42.57
```

**6 configurations testées :** 20 départs · max_iter = 3000 · `seed=42`

| Configuration | Coût moyen | Coût min | P(global) | Acc. dégr. | Itérations moy. |
|:---:|:---:|:---:|:---:|:---:|:---:|
| T₀/2, λ=0.90 | -19.3750 | -19.5000 | 75% | 12.58% | 162 |
| T₀,   λ=0.90 | -19.3000 | -19.5000 | 60% | 15.12% | 168 |
| 2·T₀, λ=0.90 | -19.3250 | -19.5000 | 65% | 18.39% | 175 |
| T₀,   λ=0.85 | -19.2250 | -19.5000 | 45% | 17.01% | 110 |
| T₀,   λ=0.95 | -19.3500 | -19.5000 | 70% | 13.99% | 344 |
| **T₀, λ=0.99** | **-19.5000** | **-19.5000** | **100%** | 13.23% | **1749** |

> ⚡ **λ=0.85** : refroidissement trop rapide → exploration insuffisante → 55% d'échecs  
> ⏳ **λ=0.99** : 100% de succès mais ~**×10** en temps de calcul

| Comparaison des configurations | Évolution température + coût |
|:---:|:---:|
| ![Recuit comp](figures/A3_recuit_comparaison.png) | ![Recuit évol](figures/A3_recuit_evolution.png) |

> **Conclusion A.3 :** Le recuit simulé est le plus robuste à l'initialisation parmi les méthodes à solution unique. La qualité de T₀ estimée empiriquement est validée expérimentalement. Le compromis exploration/exploitation est entièrement contrôlé par λ.

---

## 🔶 Volet B — Algorithme génétique

**Fonction objectif à maximiser :**

$$f(x) = \sin(x) \cdot e^{\sin(x)}, \quad x \in [-5,\; 5]$$

**Maximum global :** $f^* = e \approx 2.71828$ atteint en $x^* = \pi/2 \approx 1.5708$ (et par symétrie en $x \approx -3\pi/2 \approx -4.712$).

![Fonction fitness](figures/B0_fonction.png)

---

### B.1 — Codage binaire

**Représentation génotypique :** chromosomes de **N = 10 bits** → phénotype réel x ∈ [-5, 5]

**Décodage (génotype → phénotype) :**

$$x = x_{\min} + \text{décimal}(c) \cdot \frac{x_{\max} - x_{\min}}{2^N - 1} = -5 + \text{décimal}(c) \cdot \frac{10}{1023}$$

**Précision du codage :**

$$\Delta x = \frac{x_{\max} - x_{\min}}{2^N - 1} = \frac{10}{1023} \approx 0.00977$$

**Exemples de chromosomes :**

| Chromosome (10 bits) | Décimal | x | f(x) | Note |
|:---:|:---:|:---:|:---:|---|
| `0000000000` | 0 | -5.0000 | +2.5017 | Bord gauche |
| `1111111111` | 1023 | +5.0000 | -0.3676 | Bord droit |
| `0101010101` | 341 | -1.6667 | -0.3679 | Intermédiaire |
| `1000000000` | 512 | +0.0049 | +0.0049 | ≈ Origine |

---

### B.2 — Opérateurs génétiques

#### Sélection

| Méthode | Principe | Avantage |
|---|---|---|
| **Roulette** | Probabilité ∝ fitness — `p(i) = f(i) / Σf` | Douce, préserve la diversité |
| **Tournoi** (k=3) | Choisit le meilleur parmi k candidats aléatoires | Plus sélective, convergence plus rapide |

#### Croisement bipoints

```
Parent 1 :  [1 0 1 | 0 1 0 | 1 1 0 1]
Parent 2 :  [0 1 0 | 1 0 1 | 0 0 1 0]
                  p₁         p₂
Enfant 1 :  [1 0 1 | 1 0 1 | 1 1 0 1]   ← segments de P1, P2, P1
Enfant 2 :  [0 1 0 | 0 1 0 | 0 0 1 0]   ← segments de P2, P1, P2
```

Croisement appliqué avec probabilité **pc** (sinon copie directe).

#### Mutation bit à bit

Chaque bit est inversé indépendamment avec probabilité **pm** :

```
Avant  :  [1  0  1  0  1  1  0  1  0  1]
pm=0.01:   .  .  .  .  .  ↕  .  .  .  .   ← bit 5 muté
Après  :  [1  0  1  0  1  0  0  1  0  1]
```

#### Élitisme

Le meilleur individu de chaque génération est **automatiquement copié** dans la génération suivante — garantit la non-régression de la meilleure solution.

**8 configurations expérimentées :**

| Configuration | x trouvé | f(x) | Erreur vs f* |
|:---:|:---:|:---:|:---:|
| pop=20, pm=0.01, roulette  | -4.7067 | 2.718195 | 8.65×10⁻⁵ |
| pop=50, pm=0.01, roulette  | -4.7165 | 2.718235 | 4.63×10⁻⁵ |
| **pop=100, roulette**      | **1.5689** | **2.718272** | **9.57×10⁻⁶** |
| **pop=50, tournoi**        | **1.5689** | **2.718272** | **9.57×10⁻⁶** |
| pop=50, pm=0.05, roulette  | -4.7165 | 2.718235 | 4.63×10⁻⁵ |
| pop=50, pm=0.10, roulette  | 1.5689  | 2.718272 | 9.57×10⁻⁶ |
| pop=50, pc=0.60, roulette  | 1.5689  | 2.718272 | 9.57×10⁻⁶ |
| pop=50, pc=0.90, roulette  | 1.5689  | 2.718272 | 9.57×10⁻⁶ |

> ⚠️ **pop=20** : population trop petite → dérive génétique → coincé dans le bassin local `x ≈ -4.71` (2ème maximum local)

| Convergence par génération | Effet du taux de mutation |
|:---:|:---:|
| ![AG conv](figures/B2_ag_convergence.png) | ![AG mut](figures/B2_ag_mutation.png) |

---

### B.3 — Analyse des schèmes de Holland

Le **théorème fondamental des schèmes** (Holland, 1975) stipule que les schèmes courts, d'ordre faible et de haute fitness se propagent **exponentiellement** dans la population.

**Probabilité de destruction d'un schème H :**

$$P_{dest}(H) = 1 - \underbrace{\left(1 - p_c \cdot \frac{u(H)}{N-1}\right)}_{\text{survie croisement}} \cdot \underbrace{\left(1 - p_m\right)^{o(H)}}_{\text{survie mutation}}$$

où : `o(H)` = ordre (bits définis), `u(H)` = longueur utile (distance premier–dernier bit défini)

**Analyse de 4 schèmes (pc=0.8, pm=0.01) :**

| Schème H | o(H) | u(H) | P(destruction) | Survie attendue |
|:---:|:---:|:---:|:---:|---|
| `1*********` | 1 | 0 | **0.010** | 🟢 Très robuste — bit₀=1 → x ∈ [0,5] |
| `11********` | 2 | 1 | 0.107 | 🟡 Robuste — deux bits → x > 2.5 |
| `****1111**` | 4 | 3 | 0.296 | 🟠 Intermédiaire |
| `1000****1*` | 5 | 8 | **0.725** | 🔴 Fragile — schème long, rarement conservé |

**Évolution de la présence des schèmes dans la population :**

| Génération | \|H₁\| (`1*…`) | \|H₄\| (`1000****1*`) | Fitness moy. |
|:---:|:---:|:---:|:---:|
| 1 | 24 | 2 | 0.842 |
| 10 | 5 | 1 | 2.514 |
| 20 | 4 | 1 | 2.370 |
| 40 | 1 | 0 | 2.525 |
| 60 | 0 | 0 | 2.530 |
| 100 | 1 | 1 | 2.558 |

> **Vérification expérimentale :** H₄ (u=8, P_dest=72%) ne survit jamais durablement. H₁ (u=0, P_dest=1%) persiste longtemps mais disparaît finalement par dérive génétique — **confirmation du théorème de Holland**.

---

## 📊 Comparaison globale

### Volet A — Problème binaire · f* = -19.5000

| Méthode | Coût moyen | Coût min | P(global) | Itérations moy. | Temps relatif |
|---|:---:|:---:|:---:|:---:|:---:|
| Descente locale | -19.3250 | -19.5000 | 65% | ~4 | ⚡ ×1 |
| Taboue k=5 (Mvt) | **-19.5000** | **-19.5000** | **100%** | 300 | ×75 |
| Recuit (T₀, λ=0.90) | -19.3750 | -19.5000 | 75% | 168 | ×42 |
| Recuit (T₀, λ=0.99) | **-19.5000** | **-19.5000** | **100%** | 1749 | ×437 |

### Volet B — Algorithme génétique · f* ≈ 2.71828

| Configuration | Fitness trouvée | Erreur absolue | Convergence |
|---|:---:|:---:|:---:|
| pop=20, roulette, pm=0.01 | 2.718195 | 8.7×10⁻⁵ | Lente, bassin local |
| pop=50, roulette, pm=0.01 | 2.718235 | 4.6×10⁻⁵ | Correcte |
| pop=50, tournoi, pm=0.01 | **2.718272** | **9.6×10⁻⁶** | Rapide |
| pop=100, roulette, pm=0.01 | **2.718272** | **9.6×10⁻⁶** | Rapide |

![Comparaison globale](figures/C_comparaison_globale.png)

---

## 🔬 Discussion scientifique

| Méthode | ✅ Forces | ❌ Limites | 📌 Contexte d'usage |
|---|---|---|---|
| **Descente locale** | Ultra-rapide · Déterministe | Piégée dans les minima locaux (35%) · Totalement dépendante de l'initialisation | Borne de référence basse · Composante d'une méthode hybride |
| **Recherche taboue** | 100% de succès dès k≥3 · Mémoire force la diversification · Critère d'aspiration intelligent | k trop grand → blocage si n petit · Mémoire additionnelle | Espaces discrets bien structurés · Complémentaire à la descente |
| **Recuit simulé** | Très robuste à l'initialisation · T₀ estimable empiriquement · Paramètres interprétables | λ=0.99 : ×10 en temps · Sensible au calendrier de refroidissement | Surface d'énergie inconnue · Problèmes difficiles à modéliser |
| **Algorithme génétique** | Applicabilité naturelle à l'espace continu · Exploration parallèle · Opérateurs évolutionnaires | Convergence prématurée si pop trop petite · Nombreux hyperparamètres | Fonctions multimodales continues · Problèmes boîte noire |

> **Conclusion générale :**  
> Aucune méthode n'est universellement supérieure. La recherche taboue excelle sur l'espace discret binaire (100% de succès, mémoire efficace). Le recuit simulé offre le meilleur compromis robustesse/généralité. L'algorithme génétique est indispensable pour l'espace continu et la recherche à population.

---

## 🚀 Perspectives

| # | Perspective | Impact attendu |
|:---:|---|---|
| 1 | **Hybridation AG + Taboue** | Initialiser l'AG avec des solutions de la recherche taboue pour combiner diversification globale et intensification locale | Forte amélioration qualité |
| 2 | **Adaptation dynamique** | Ajuster λ (recuit) ou k (taboue) selon la progression de la recherche | Réduction ×2 à ×3 du temps |
| 3 | **Passage à l'échelle** | Tester sur n=50, n=100 bits pour étudier la complexité empirique | Validation expérimentale |
| 4 | **Validation statistique** | Tests de Wilcoxon/Mann-Whitney sur 100+ exécutions | Rigueur scientifique |
| 5 | **Baseline aléatoire** | Ajouter Random Restart comme référence basse | Contextualisation des gains |

---

## 📚 Références

- Holland, J.H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Glover, F. (1989). *Tabu Search — Part I*. ORSA Journal on Computing, **1**(3), 190–206.
- Kirkpatrick, S., Gelatt, C.D., Vecchi, M.P. (1983). *Optimization by Simulated Annealing*. Science, **220**(4598), 671–680.
- Goldberg, D.E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.

---

<div align="center">

---

[![Site interactif](https://img.shields.io/badge/🌐_Site_de_simulation-Ouvrir-blue?style=for-the-badge)](https://hichamouaouche.github.io/M-taheuristiques_et_algorithmes_-_base_de_population/)
[![Dépôt GitHub](https://img.shields.io/badge/📁_Dépôt_GitHub-Voir-222?style=for-the-badge&logo=github)](https://github.com/hichamouaouche/M-taheuristiques_et_algorithmes_-_base_de_population)

*Projet réalisé dans le cadre du Master – Optimisation et Recherche Opérationnelle · ENSET · 2025-2026*

</div>
