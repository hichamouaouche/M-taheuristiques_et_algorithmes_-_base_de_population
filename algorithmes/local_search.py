"""
local_search.py – A.1 : Descente locale (steepest descent)

Algorithme :
  - À chaque itération, choisir le meilleur voisin admissible (Hamming-1).
  - S'arrêter si aucun voisin n'améliore la solution courante.
  - Tester 20 initialisations aléatoires différentes.

Analyses :
  a) Minima locaux atteints
  b) Probabilité empirique d'atteindre le minimum global
  c) Sensibilité à l'initialisation
"""

import numpy as np
from .problem import BinaryProblem


def steepest_descent(problem, initial_solution, verbose=False):
    """
    Descente selon la plus grande pente.

    Retourne :
        best_sol   : meilleure solution trouvée
        best_cost  : coût de cette solution
        history    : liste des coûts à chaque itération
        n_iter     : nombre d'itérations effectuées
    """
    s = list(initial_solution)
    c = problem.cost(s)
    history = [c]

    iteration = 0
    while True:
        neighbors = problem.neighbors(s)
        costs = [problem.cost(nb) for nb in neighbors]

        best_nb_idx = int(np.argmin(costs))
        best_nb_cost = costs[best_nb_idx]

        if best_nb_cost >= c:
            # Aucune amélioration possible → minimum local atteint
            break

        s = neighbors[best_nb_idx]
        c = best_nb_cost
        history.append(c)
        iteration += 1

        if verbose:
            print(f"  Iter {iteration:3d} : f = {c:.4f}  s = {problem.solution_str(s)}")

    return s, c, history, iteration


def run_local_search(problem, n_starts=20, seed=42):
    """
    Lance la descente locale depuis n_starts solutions initiales différentes.
    Retourne des statistiques et tous les résultats.
    """
    rng = np.random.default_rng(seed)
    _, global_opt, _ = problem.brute_force()

    results = []
    for i in range(n_starts):
        s0 = problem.random_solution(rng)
        sol, cost, history, n_iter = steepest_descent(problem, s0)
        is_global = abs(cost - global_opt) < 1e-6
        results.append({
            'start': i + 1,
            'init_sol': problem.solution_str(s0),
            'init_cost': problem.cost(s0),
            'final_sol': problem.solution_str(sol),
            'final_cost': cost,
            'n_iter': n_iter,
            'is_global': is_global,
            'history': history,
        })

    # Statistiques globales
    n_global = sum(r['is_global'] for r in results)
    prob_global = n_global / n_starts
    final_costs = [r['final_cost'] for r in results]
    local_minima_found = {}
    for r in results:
        k = r['final_sol']
        if k not in local_minima_found:
            local_minima_found[k] = {'cost': r['final_cost'], 'count': 0}
        local_minima_found[k]['count'] += 1

    stats = {
        'n_starts': n_starts,
        'n_global': n_global,
        'prob_global': prob_global,
        'mean_final_cost': float(np.mean(final_costs)),
        'std_final_cost': float(np.std(final_costs)),
        'best_cost': float(np.min(final_costs)),
        'local_minima': sorted(local_minima_found.items(), key=lambda x: x[1]['cost']),
    }

    return results, stats


def print_local_search_report(results, stats, global_opt_cost):
    """Affiche le rapport de la descente locale."""
    print("\n" + "=" * 65)
    print("A.1 – DESCENTE LOCALE (Steepest Descent)")
    print("=" * 65)

    print(f"\n{'#':>3} | {'Init':>12} | {'Final':>12} | {'Coût init':>10} | "
          f"{'Coût final':>10} | {'Iter':>5} | {'Global?':>7}")
    print("-" * 75)
    for r in results:
        flag = "✓" if r['is_global'] else " "
        print(f"{r['start']:>3} | {r['init_sol']:>12} | {r['final_sol']:>12} | "
              f"{r['init_cost']:>10.4f} | {r['final_cost']:>10.4f} | "
              f"{r['n_iter']:>5} | {flag:>7}")

    print("\n--- Analyse ---")
    print(f"a) Minima locaux atteints :")
    for sol_str, info in stats['local_minima']:
        marker = " ← MIN. GLOBAL" if abs(info['cost'] - global_opt_cost) < 1e-6 else ""
        print(f"   s = {sol_str}  f = {info['cost']:8.4f}  "
              f"atteint {info['count']:2d} fois{marker}")

    print(f"\nb) Probabilité empirique d'atteindre le minimum global :")
    print(f"   P(global) = {stats['n_global']}/{stats['n_starts']} = "
          f"{stats['prob_global']:.2%}")

    print(f"\nc) Sensibilité à l'initialisation :")
    print(f"   Coût final moyen    = {stats['mean_final_cost']:.4f}")
    print(f"   Écart-type          = {stats['std_final_cost']:.4f}")
    print(f"   Meilleur coût final = {stats['best_cost']:.4f}")
    print(f"   → La méthode est {'peu' if stats['std_final_cost'] < 1 else 'très'} "
          f"sensible à l'initialisation.")


# ------------------------------------------------------------------
if __name__ == '__main__':
    prob = BinaryProblem()
    print("Calcul du minimum global par force brute...")
    best_sol, best_cost, local_mins = prob.brute_force()
    print(f"Minimum global : f* = {best_cost:.4f}  s* = {prob.solution_str(best_sol)}")

    results, stats = run_local_search(prob, n_starts=20, seed=42)
    print_local_search_report(results, stats, best_cost)
