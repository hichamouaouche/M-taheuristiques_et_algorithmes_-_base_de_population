

import numpy as np
from .problem import BinaryProblem


def estimate_T0(problem, n_samples=500, acceptance_rate=0.80, seed=0):

    rng = np.random.default_rng(seed)
    delta_E_list = []

    for _ in range(n_samples):
        s = problem.random_solution(rng)
        c = problem.cost(s)
        neighbors = problem.neighbors(s)
        for nb in neighbors:
            delta = problem.cost(nb) - c
            if delta > 0:
                delta_E_list.append(delta)

    if not delta_E_list:
        return 1.0

    delta_max = float(np.percentile(delta_E_list, 95))  # percentile 95 pour robustesse
    T0 = -delta_max / np.log(acceptance_rate)
    return T0


def simulated_annealing(problem, initial_solution, T0, lam, max_iter=2000,
                         T_min=1e-6, verbose=False):
    """
    Recuit simulé.

    Paramètres :
        T0       : température initiale
        lam      : facteur de refroidissement (λ)
        max_iter : nombre maximal d'itérations
        T_min    : température minimale (arrêt)

    Retourne :
        best_sol, best_cost, history (coût à chaque itération),
        T_history (log des températures)
    """
    rng = np.random.default_rng()
    s = list(initial_solution)
    c = problem.cost(s)
    best_s = s.copy()
    best_c = c

    T = T0
    history = [c]
    T_history = [T]
    n_accepted = 0
    n_degrading = 0

    for iteration in range(max_iter):
        if T < T_min:
            break

        # Choisir un voisin aléatoire (flip d'un bit au hasard)
        i_flip = int(rng.integers(0, problem.n))
        nb = s.copy()
        nb[i_flip] = 1 - nb[i_flip]
        nb_c = problem.cost(nb)

        delta_E = nb_c - c

        if delta_E <= 0:
            # Amélioration → acceptation automatique
            s = nb
            c = nb_c
            n_accepted += 1
        else:
            # Dégradation → acceptation probabiliste
            n_degrading += 1
            prob_accept = np.exp(-delta_E / T)
            if rng.random() < prob_accept:
                s = nb
                c = nb_c
                n_accepted += 1

        if c < best_c:
            best_c = c
            best_s = s.copy()

        # Refroidissement
        T = lam * T
        history.append(c)
        T_history.append(T)

        if verbose and (iteration % 200 == 0):
            print(f"  Iter {iteration:5d}: T={T:.4f}  f={c:.4f}  best={best_c:.4f}")

    return best_s, best_c, history, T_history, n_accepted, n_degrading


def run_simulated_annealing(problem, configs=None, max_iter=3000, n_starts=15, seed=42):
    """
    Compare plusieurs couples (T₀, λ).

    configs : liste de (T0, lambda), si None → configs par défaut
    """
    if configs is None:
        # Estimation automatique de T₀
        T0_auto = estimate_T0(problem, seed=seed)
        T0_half = T0_auto / 2
        T0_double = T0_auto * 2
        configs = [
            (T0_half,   0.90, f"T0/2,  λ=0.90"),
            (T0_auto,   0.90, f"T0,    λ=0.90"),
            (T0_double, 0.90, f"2·T0,  λ=0.90"),
            (T0_auto,   0.85, f"T0,    λ=0.85"),
            (T0_auto,   0.95, f"T0,    λ=0.95"),
            (T0_auto,   0.99, f"T0,    λ=0.99"),
        ]

    rng = np.random.default_rng(seed)
    _, global_opt, _ = problem.brute_force()

    all_results = {}

    for T0, lam, label in configs:
        runs = []
        for _ in range(n_starts):
            s0 = problem.random_solution(rng)
            best_s, best_c, hist, T_hist, n_acc, n_deg = simulated_annealing(
                problem, s0, T0=T0, lam=lam, max_iter=max_iter)
            runs.append({
                'best_cost': best_c,
                'best_sol': problem.solution_str(best_s),
                'history': hist,
                'T_history': T_hist,
                'n_accepted': n_acc,
                'n_degrading': n_deg,
                'is_global': abs(best_c - global_opt) < 1e-6,
            })
        all_results[label] = {'T0': T0, 'lam': lam, 'runs': runs}

    return all_results, global_opt


def print_sa_report(all_results, global_opt):
    """Affiche le rapport du recuit simulé."""
    print("\n" + "=" * 70)
    print("A.3 – RECUIT SIMULÉ")
    print("=" * 70)

    T0_auto = None
    for label, res in all_results.items():
        if 'T0,    λ=0.90' in label:
            T0_auto = res['T0']
            break
    if T0_auto:
        print(f"\nTempérature initiale estimée T₀ = {T0_auto:.4f}")
        print(f"  (basée sur percentile 95 des ΔE > 0, cible P(accept dégradation)≈80%)")

    print(f"\n{'Configuration':>20} | {'Coût moy':>10} | {'Coût min':>10} | "
          f"{'P(global)':>10} | {'Acc. dégr.':>11} | {'Itérations':>11}")
    print("-" * 85)

    for label, res in all_results.items():
        runs = res['runs']
        best_costs = [r['best_cost'] for r in runs]
        n_global = sum(r['is_global'] for r in runs)
        acc_rates = [r['n_accepted'] / (r['n_accepted'] + r['n_degrading'])
                     if (r['n_accepted'] + r['n_degrading']) > 0 else 0
                     for r in runs]
        avg_iters = np.mean([len(r['history']) for r in runs])

        print(f"{label:>20} | {np.mean(best_costs):>10.4f} | {np.min(best_costs):>10.4f} | "
              f"{n_global/len(runs):>10.2%} | {np.mean(acc_rates):>11.2%} | "
              f"{avg_iters:>11.0f}")

    print("\n--- Analyse ---")
    print("• T₀ trop bas  → ressemble à une descente locale (peu d'échappements)")
    print("• T₀ trop haut → exploration trop aléatoire au début")
    print("• λ proche de 1 → refroidissement lent, meilleure qualité mais plus lent")
    print("• λ proche de 0.85 → convergence rapide mais risque de rater le global")
    print("• L'effet de T sur l'acceptation de dégradations :")
    print("  – T élevé → P(accept) ≈ 1 même pour de grandes dégradations")
    print("  – T bas   → seules les très petites dégradations sont acceptées")


# ------------------------------------------------------------------
if __name__ == '__main__':
    prob = BinaryProblem()
    T0_est = estimate_T0(prob)
    print(f"Température initiale estimée : T₀ = {T0_est:.4f}")

    all_res, global_opt = run_simulated_annealing(prob, max_iter=3000,
                                                   n_starts=20, seed=42)
    print_sa_report(all_res, global_opt)
    print(f"\nMinimum global de référence : f* = {global_opt:.4f}")
