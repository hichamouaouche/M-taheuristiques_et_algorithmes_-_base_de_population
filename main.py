

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   # pas besoin d'un écran graphique

from algorithmes.problem import BinaryProblem
from algorithmes.local_search import run_local_search, print_local_search_report
from algorithmes.tabu_search import run_tabu_search, print_tabu_report
from algorithmes.simulated_annealing import run_simulated_annealing, print_sa_report, estimate_T0
from algorithmes.genetic_algorithm import (
    run_ga_experiments, print_ga_report, analyze_schemas,
    show_coding_examples, decode, fitness as ga_fitness
)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# ================================================================
def save(fig, name):
    path = os.path.join(FIGURES_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  → figure sauvegardée : figures/{name}")


# ================================================================
# SECTION A – PROBLÈME BINAIRE
# ================================================================

def run_volet_A():
    prob = BinaryProblem()

    print("\n" + "#" * 70)
    print("# VOLET A – RECHERCHE LOCALE ET MÉTAHEURISTIQUES DE VOISINAGE")
    print("#" * 70)

    # --- Force brute ---
    print("\nForce brute (2^10 = 1024 solutions) ...")
    best_sol, best_cost, local_mins = prob.brute_force()
    print(f"  Minimum global : s* = {prob.solution_str(best_sol)}  f* = {best_cost:.4f}")
    print(f"  Nombre de minima locaux : {len(local_mins)}")

    # Figure : distribution des coûts
    all_costs = [prob.cost([int(b) for b in format(i, f'0{prob.n}b')])
                 for i in range(2 ** prob.n)]
    local_min_costs = [c for _, c in local_mins]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(all_costs, bins=40, color='steelblue', edgecolor='white', alpha=0.8,
            label='Toutes solutions')
    ax.scatter(local_min_costs, [2] * len(local_min_costs),
               color='red', zorder=5, s=60, label='Minima locaux')
    ax.axvline(best_cost, color='darkred', lw=2, linestyle='--',
               label=f'Min. global f*={best_cost:.2f}')
    ax.set_xlabel('Coût f(s)')
    ax.set_ylabel('Nombre de solutions')
    ax.set_title("Distribution des coûts – Espace binaire {0,1}^10")
    ax.legend()
    save(fig, "A0_distribution_couts.png")

    # --------------------------------------------------------
    # A.1 – Descente locale
    # --------------------------------------------------------
    print("\n--- A.1 Descente locale ---")
    ls_results, ls_stats = run_local_search(prob, n_starts=20, seed=42)
    print_local_search_report(ls_results, ls_stats, best_cost)

    # Figure : évolution du coût pour chaque départ
    fig, ax = plt.subplots(figsize=(9, 5))
    for r in ls_results:
        color = 'green' if r['is_global'] else 'salmon'
        ax.plot(r['history'], color=color, alpha=0.6, lw=1.2)
    ax.axhline(best_cost, color='black', lw=2, linestyle='--',
               label=f'Min. global f*={best_cost:.2f}')
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='green', lw=2, label='Atteint le global'),
        Line2D([0], [0], color='salmon', lw=2, label='Coincé dans un local'),
        Line2D([0], [0], color='black', lw=2, ls='--', label='Min. global'),
    ]
    ax.legend(handles=legend_elements)
    ax.set_xlabel('Itération')
    ax.set_ylabel('Coût f(s)')
    ax.set_title("A.1 – Descente locale : évolution du coût (20 départs)")
    save(fig, "A1_descente_locale.png")

    # --------------------------------------------------------
    # A.2 – Recherche taboue
    # --------------------------------------------------------
    print("\n--- A.2 Recherche Taboue ---")
    tabu_results, _ = run_tabu_search(prob, tabu_sizes=(1, 3, 5, 10),
                                       max_iter=300, n_starts=20, seed=42)
    print_tabu_report(tabu_results, best_cost)

    # Figure 1 : comparaison des coûts moyens par k et stratégie
    ks = [1, 3, 5, 10]
    means_sol = [np.mean([r['best_cost'] for r in tabu_results[k]['solutions']]) for k in ks]
    means_mov = [np.mean([r['best_cost'] for r in tabu_results[k]['moves']]) for k in ks]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, means_sol, 'o-', label='Stratégie Solutions', color='steelblue')
    ax.plot(ks, means_mov, 's--', label='Stratégie Mouvements', color='darkorange')
    ax.axhline(best_cost, color='black', lw=1.5, linestyle=':', label=f'f*={best_cost:.2f}')
    ax.set_xlabel('Taille liste taboue k')
    ax.set_ylabel('Coût moyen final')
    ax.set_title("A.2 – Recherche Taboue : effet de k et de la stratégie")
    ax.legend()
    ax.set_xticks(ks)
    save(fig, "A2_tabu_k_strategie.png")

    # Figure 2 : exemples d'évolution (k=5, stratégie mouvements)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, r in enumerate(tabu_results[5]['moves'][:5]):
        ax.plot(r['history'], alpha=0.7, lw=1.2, label=f'Essai {i+1}')
    ax.axhline(best_cost, color='black', lw=2, linestyle='--',
               label=f'Min. global f*={best_cost:.2f}')
    ax.set_xlabel('Itération')
    ax.set_ylabel('Coût f(s)')
    ax.set_title("A.2 – Recherche Taboue (k=5, stratégie Mouvements)")
    ax.legend(fontsize=8)
    save(fig, "A2_tabu_evolution.png")

    # --------------------------------------------------------
    # A.3 – Recuit simulé
    # --------------------------------------------------------
    print("\n--- A.3 Recuit Simulé ---")
    T0_est = estimate_T0(prob, seed=42)
    sa_results, _ = run_simulated_annealing(prob, max_iter=3000, n_starts=20, seed=42)
    print_sa_report(sa_results, best_cost)

    # Figure 1 : comparaison coûts finaux par configuration
    labels_sa = list(sa_results.keys())
    means_sa = [np.mean([r['best_cost'] for r in sa_results[lb]['runs']]) for lb in labels_sa]
    mins_sa = [np.min([r['best_cost'] for r in sa_results[lb]['runs']]) for lb in labels_sa]

    fig, ax = plt.subplots(figsize=(10, 4))
    x_pos = np.arange(len(labels_sa))
    ax.bar(x_pos - 0.2, means_sa, width=0.35, label='Coût moyen', color='steelblue')
    ax.bar(x_pos + 0.2, mins_sa, width=0.35, label='Coût min', color='darkorange')
    ax.axhline(best_cost, color='black', lw=1.5, linestyle='--', label=f'f*={best_cost:.2f}')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels_sa, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Coût final')
    ax.set_title("A.3 – Recuit Simulé : comparaison de configurations (T₀, λ)")
    ax.legend()
    save(fig, "A3_recuit_comparaison.png")

    # Figure 2 : évolution de la température et du coût pour une config typique
    label_ref = list(sa_results.keys())[1]  # T0, λ=0.90
    ref_run = sa_results[label_ref]['runs'][0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(ref_run['T_history'], color='darkorange', lw=1.2)
    ax1.set_ylabel('Température T')
    ax1.set_title(f"A.3 – Recuit Simulé ({label_ref}) : température et coût")
    ax2.plot(ref_run['history'], color='steelblue', lw=1.2)
    ax2.axhline(best_cost, color='red', lw=1.5, linestyle='--',
                label=f'f*={best_cost:.2f}')
    ax2.set_xlabel('Itération')
    ax2.set_ylabel('Coût f(s)')
    ax2.legend()
    plt.tight_layout()
    save(fig, "A3_recuit_evolution.png")

    return prob, best_cost, ls_stats, tabu_results, sa_results


# ================================================================
# SECTION B – ALGORITHME GÉNÉTIQUE
# ================================================================

def run_volet_B():
    print("\n" + "#" * 70)
    print("# VOLET B – ALGORITHME GÉNÉTIQUE (OPTIMISATION CONTINUE)")
    print("#" * 70)

    # B.1 – Codage
    show_coding_examples()

    # Figure : la fonction f(x)
    xs = np.linspace(-5, 5, 1000)
    ys = np.sin(xs) * np.exp(np.sin(xs))
    best_x_true = float(xs[np.argmax(ys)])
    best_y_true = float(np.max(ys))

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(xs, ys, lw=2, color='steelblue')
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(best_x_true, color='red', lw=1.5, linestyle='--',
               label=f'max : x={best_x_true:.3f}, f={best_y_true:.4f}')
    ax.scatter([best_x_true], [best_y_true], color='red', s=80, zorder=5)
    ax.set_xlabel('x')
    ax.set_ylabel('f(x) = sin(x)·exp(sin(x))')
    ax.set_title("Volet B – Fonction à maximiser")
    ax.legend()
    save(fig, "B0_fonction.png")

    # B.2 – Expériences AG
    print("\n--- B.2 Opérateurs génétiques ---")
    ga_results = run_ga_experiments()
    print_ga_report(ga_results)

    # Figure 1 : convergence des meilleurs individus pour plusieurs configs
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    configs_to_plot = [
        'pop=20,  pm=0.01, roulette',
        'pop=50,  pm=0.01, roulette',
        'pop=100, pm=0.01, roulette',
        'pop=50,  pm=0.01, tournoi',
    ]
    for ax, label in zip(axes, configs_to_plot):
        res = ga_results[label]
        ax.plot(res['best_hist'], lw=2, label='Meilleure fitness', color='steelblue')
        ax.plot(res['mean_hist'], lw=1.5, linestyle='--',
                label='Fitness moyenne', color='darkorange')
        ax.fill_between(range(len(res['mean_hist'])),
                        np.array(res['mean_hist']) - np.array(res['div_hist']),
                        np.array(res['mean_hist']) + np.array(res['div_hist']),
                        alpha=0.2, color='darkorange')
        ax.axhline(best_y_true, color='red', lw=1.5, linestyle=':', label=f'max réel')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('Génération')
        ax.set_ylabel('Fitness')
        ax.legend(fontsize=7)
    plt.suptitle("B.2 – Évolution de la fitness par génération", fontsize=12)
    plt.tight_layout()
    save(fig, "B2_ag_convergence.png")

    # Figure 2 : effet du taux de mutation
    fig, ax = plt.subplots(figsize=(9, 4))
    for pm_label in ['pop=50,  pm=0.01, roulette',
                      'pop=50,  pm=0.05, roulette',
                      'pop=50,  pm=0.10, roulette']:
        res = ga_results[pm_label]
        pm_val = pm_label.split('pm=')[1].split(',')[0]
        ax.plot(res['best_hist'], lw=2, label=f'pm={pm_val}')
    ax.axhline(best_y_true, color='black', lw=1.5, linestyle='--', label='max réel')
    ax.set_xlabel('Génération')
    ax.set_ylabel('Meilleure fitness')
    ax.set_title("B.2 – Effet du taux de mutation pm")
    ax.legend()
    save(fig, "B2_ag_mutation.png")

    # B.3 – Analyse des schèmes
    print("\n--- B.3 Analyse des schèmes ---")
    analyze_schemas()

    return ga_results


# ================================================================
# SECTION C – COMPARAISON GLOBALE
# ================================================================

def run_comparaison(best_cost_binary, ls_stats, tabu_results, sa_results, ga_results):
    print("\n" + "#" * 70)
    print("# 3 – COMPARAISON GLOBALE DES MÉTHODES")
    print("#" * 70)

    # --------- Volet A : comparaison sur problème binaire ----------
    # Résumé tabu (k=5, mouvements)
    tabu_runs = tabu_results[5]['moves']
    tabu_mean = np.mean([r['best_cost'] for r in tabu_runs])
    tabu_min = np.min([r['best_cost'] for r in tabu_runs])
    tabu_globe = sum(r['is_global'] for r in tabu_runs) / len(tabu_runs)

    # Résumé recuit (config centrale : T0, λ=0.90)
    sa_key = list(sa_results.keys())[1]
    sa_runs = sa_results[sa_key]['runs']
    sa_mean = np.mean([r['best_cost'] for r in sa_runs])
    sa_min = np.min([r['best_cost'] for r in sa_runs])
    sa_globe = sum(r['is_global'] for r in sa_runs) / len(sa_runs)

    print("\n── Tableau synthétique – Problème binaire (min. global = {:.4f}) ──".format(
        best_cost_binary))
    print(f"\n{'Méthode':<25} | {'Coût moy':>10} | {'Coût min':>10} | "
          f"{'P(global)':>10} | Notes")
    print("-" * 80)
    print(f"{'Descente locale':<25} | "
          f"{ls_stats['mean_final_cost']:>10.4f} | {ls_stats['best_cost']:>10.4f} | "
          f"{ls_stats['prob_global']:>10.2%} | Rapide, coincé souvent")
    print(f"{'Taboue k=5 (Mouvements)':<25} | "
          f"{tabu_mean:>10.4f} | {tabu_min:>10.4f} | "
          f"{tabu_globe:>10.2%} | Meilleure diversification")
    print(f"{'Recuit simulé':<25} | "
          f"{sa_mean:>10.4f} | {sa_min:>10.4f} | "
          f"{sa_globe:>10.2%} | Robuste, lent si λ → 1")

    # --------- Volet B : comparaison AG ----------
    ga_ref = ga_results['pop=50,  pm=0.01, roulette']
    xs = np.linspace(-5, 5, 10000)
    true_max = float(np.max(np.sin(xs) * np.exp(np.sin(xs))))

    print(f"\n── Tableau synthétique – Fonction continue (max réel = {true_max:.6f}) ──")
    print(f"\n{'Configuration AG':<40} | {'f(x) trouvé':>12} | {'Erreur':>10}")
    print("-" * 70)
    for label, res in ga_results.items():
        err = abs(res['best_f'] - true_max)
        print(f"{label:<40} | {res['best_f']:>12.6f} | {err:>10.6f}")

    # ---------- Figure de comparaison générale ----------
    methods = ['Descente\nlocale', 'Taboue\nk=5', 'Recuit\nsimulé']
    means_A = [ls_stats['mean_final_cost'], tabu_mean, sa_mean]
    mins_A = [ls_stats['best_cost'], tabu_min, sa_min]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Volet A
    x_pos = np.arange(len(methods))
    axes[0].bar(x_pos - 0.2, means_A, width=0.35, label='Coût moyen', color='steelblue')
    axes[0].bar(x_pos + 0.2, mins_A, width=0.35, label='Meilleur coût', color='darkorange')
    axes[0].axhline(best_cost_binary, color='black', lw=2, ls='--',
                    label=f'Optimum f*={best_cost_binary:.2f}')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(methods)
    axes[0].set_ylabel('Coût')
    axes[0].set_title("Volet A – Comparaison des méthodes\n(problème binaire)")
    axes[0].legend(fontsize=8)

    # Volet B – convergence
    for label in ['pop=50,  pm=0.01, roulette', 'pop=50,  pm=0.01, tournoi',
                  'pop=50,  pm=0.05, roulette']:
        res = ga_results[label]
        axes[1].plot(res['best_hist'], lw=2,
                     label=label.replace('pop=50,  ', ''))
    axes[1].axhline(true_max, color='black', lw=1.5, ls='--', label='Max réel')
    axes[1].set_xlabel('Génération')
    axes[1].set_ylabel('Meilleure fitness (Volet B)')
    axes[1].set_title("Volet B – Convergence AG\n(optimisation continue)")
    axes[1].legend(fontsize=8)

    plt.suptitle("Comparaison générale", fontsize=13, fontweight='bold')
    plt.tight_layout()
    save(fig, "C_comparaison_globale.png")

    print("\n--- Discussion scientifique ---")
    print("""
Descente locale :
  ✓ Très rapide, déterministe par itération.
  ✗ Fortement dépendante de l'initialisation, coincée dans les minima locaux.
  → Utile comme benchmark ou comme composant d'algorithmes hybrides.

Recherche taboue :
  ✓ Meilleure capacité d'échappement que la descente grâce à la mémoire.
  ✓ Paramètre k permet d'ajuster intensification vs. diversification.
  ✗ Sensible au choix de k ; coûteux si on stocke toute la solution.
  → Bonne alternative pour les espaces discrets avec structure connue.

Recuit simulé :
  ✓ Robuste à l'initialisation, bon en exploration (haute température).
  ✓ Flexible grâce aux paramètres T₀ et λ.
  ✗ Difficile de choisir le bon calendrier de refroidissement.
  ✗ Convergence lente avec λ proche de 1.
  → Efficace quand la surface d'énergie est mal connue a priori.

Algorithme génétique :
  ✓ Adapté aux espaces continus, parallèle (population).
  ✓ Bon mécanisme d'exploration vs. exploitation.
  ✗ Nombreux hyperparamètres (pm, pc, pop_size, n_gen).
  ✗ Coût calculatoire plus élevé.
  → Excellent pour les fonctions multimodales continues.
""")


# ================================================================
# MAIN
# ================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("   MINI-PROJET – MÉTAHEURISTIQUES ET ALGORITHMES À BASE DE POPULATION")
    print("   ENSET / Master – Optimisation et Recherche Opérationnelle")
    print("=" * 70)

    prob, best_cost, ls_stats, tabu_results, sa_results = run_volet_A()
    ga_results = run_volet_B()
    run_comparaison(best_cost, ls_stats, tabu_results, sa_results, ga_results)

    print("\n" + "=" * 70)
    print("TERMINÉ. Toutes les figures sont dans le dossier ./figures/")
    print("=" * 70)
