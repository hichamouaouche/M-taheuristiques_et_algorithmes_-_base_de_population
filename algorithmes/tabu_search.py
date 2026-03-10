"""
tabu_search.py – A.2 : Recherche Taboue

Paramètres :
  - Taille de liste taboue k ∈ {1, 3, 5, 10}
  - Deux stratégies de mémoire :
      (1) stockage des solutions taboues
      (2) stockage des mouvements inverses (index du bit flippé)
  - Critère d'aspiration : accepter un mouvement tabou si le voisin obtenu
    est meilleur que la meilleure solution jamais trouvée.
"""

import numpy as np
from collections import deque
from .problem import BinaryProblem


# ---------------------------------------------------------------
def tabu_search_solutions(problem, initial_solution, tabu_size=5,
                           max_iter=200, verbose=False):
    """
    Recherche taboue – Stratégie 1 : liste taboue de SOLUTIONS.

    La solution courante ne peut pas être revisitée si elle est dans la liste.
    """
    s = list(initial_solution)
    c = problem.cost(s)
    best_s = s.copy()
    best_c = c

    tabu_list = deque(maxlen=tabu_size)
    tabu_list.append(tuple(s))

    history = [c]
    n_moves = 0

    for iteration in range(max_iter):
        neighbors = problem.neighbors(s)
        best_nb = None
        best_nb_cost = float('inf')

        for nb in neighbors:
            nb_t = tuple(nb)
            nb_c = problem.cost(nb)
            is_tabu = nb_t in tabu_list

            # Critère d'aspiration : accepter si meilleur que le meilleur global
            if (not is_tabu) or (nb_c < best_c):
                if nb_c < best_nb_cost:
                    best_nb_cost = nb_c
                    best_nb = nb

        if best_nb is None:
            break  # tous les voisins sont tabous et aucun ne passe l'aspiration

        s = best_nb
        c = best_nb_cost
        tabu_list.append(tuple(s))
        n_moves += 1
        history.append(c)

        if c < best_c:
            best_c = c
            best_s = s.copy()

        if verbose:
            print(f"  Iter {iteration+1:3d}: f={c:.4f}  best={best_c:.4f}  "
                  f"|tabu|={len(tabu_list)}")

    return best_s, best_c, history, n_moves


# ---------------------------------------------------------------
def tabu_search_moves(problem, initial_solution, tabu_size=5,
                       max_iter=200, verbose=False):
    """
    Recherche taboue – Stratégie 2 : liste taboue de MOUVEMENTS INVERSES.

    Un mouvement = flip du bit i.
    Le mouvement inverse = flip du même bit i (retour arrière).
    Si le bit i a été flippé, le flip de i est tabou pendant k itérations.
    """
    s = list(initial_solution)
    c = problem.cost(s)
    best_s = s.copy()
    best_c = c

    tabu_list = deque(maxlen=tabu_size)  # indices de bits tabous (mouvement aller)

    history = [c]
    n_moves = 0

    for iteration in range(max_iter):
        neighbors = problem.neighbors(s)
        best_nb = None
        best_nb_cost = float('inf')
        best_nb_idx = -1

        for i, nb in enumerate(neighbors):
            nb_c = problem.cost(nb)
            is_tabu = i in tabu_list

            # Critère d'aspiration
            if (not is_tabu) or (nb_c < best_c):
                if nb_c < best_nb_cost:
                    best_nb_cost = nb_c
                    best_nb = nb
                    best_nb_idx = i

        if best_nb is None:
            break

        s = best_nb
        c = best_nb_cost
        tabu_list.append(best_nb_idx)  # interdit de revenir en arrière
        n_moves += 1
        history.append(c)

        if c < best_c:
            best_c = c
            best_s = s.copy()

        if verbose:
            print(f"  Iter {iteration+1:3d}: f={c:.4f}  best={best_c:.4f}  "
                  f"dernier flip=bit{best_nb_idx}")

    return best_s, best_c, history, n_moves


# ---------------------------------------------------------------
def run_tabu_search(problem, tabu_sizes=(1, 3, 5, 10), max_iter=200,
                    n_starts=10, seed=42):
    """
    Compare les deux stratégies pour chaque taille de liste taboue.
    Lance n_starts exécutions depuis des initialisations aléatoires.
    """
    rng = np.random.default_rng(seed)
    _, global_opt, _ = problem.brute_force()

    all_results = {}

    for k in tabu_sizes:
        results_sol = []
        results_mov = []

        for _ in range(n_starts):
            s0 = problem.random_solution(rng)

            # Stratégie 1 : solutions
            bs1, bc1, hist1, nm1 = tabu_search_solutions(
                problem, s0, tabu_size=k, max_iter=max_iter)
            results_sol.append({
                'best_cost': bc1, 'n_moves': nm1, 'history': hist1,
                'best_sol': problem.solution_str(bs1),
                'is_global': abs(bc1 - global_opt) < 1e-6,
            })

            # Stratégie 2 : mouvements
            bs2, bc2, hist2, nm2 = tabu_search_moves(
                problem, s0, tabu_size=k, max_iter=max_iter)
            results_mov.append({
                'best_cost': bc2, 'n_moves': nm2, 'history': hist2,
                'best_sol': problem.solution_str(bs2),
                'is_global': abs(bc2 - global_opt) < 1e-6,
            })

        all_results[k] = {'solutions': results_sol, 'moves': results_mov}

    return all_results, global_opt


def print_tabu_report(all_results, global_opt):
    """Affiche le rapport de la recherche taboue."""
    print("\n" + "=" * 65)
    print("A.2 – RECHERCHE TABOUE")
    print("=" * 65)

    print(f"\n{'k':>4} | {'Stratégie':>18} | {'Coût moy':>10} | "
          f"{'Coût min':>10} | {'P(global)':>10} | {'Déplacements':>13}")
    print("-" * 75)

    for k, res in all_results.items():
        for strategy_name, run_list in [('Solutions', res['solutions']),
                                         ('Mouvements', res['moves'])]:
            costs = [r['best_cost'] for r in run_list]
            n_global = sum(r['is_global'] for r in run_list)
            n_moves = [r['n_moves'] for r in run_list]
            print(f"{k:>4} | {strategy_name:>18} | "
                  f"{np.mean(costs):>10.4f} | {np.min(costs):>10.4f} | "
                  f"{n_global/len(run_list):>10.2%} | "
                  f"{np.mean(n_moves):>13.1f}")

    print("\n--- Analyse ---")
    print("• Influence de la taille k :")
    print("  – k trop petit → peu de diversification, coincé dans minima locaux")
    print("  – k trop grand → trop restrictif, ralentit la recherche")
    print("• Stratégie 'Mouvements' : mémoire légère, meilleure diversification")
    print("• Stratégie 'Solutions' : évite exactement les solutions revisitées")
    print("• Critère d'aspiration permet d'outrepasser la liste si nécessaire")


# ------------------------------------------------------------------
if __name__ == '__main__':
    prob = BinaryProblem()
    all_results, global_opt = run_tabu_search(prob, tabu_sizes=(1, 3, 5, 10),
                                               max_iter=300, n_starts=20, seed=42)
    print_tabu_report(all_results, global_opt)
    print(f"\nMinimum global de référence : f* = {global_opt:.4f}")
