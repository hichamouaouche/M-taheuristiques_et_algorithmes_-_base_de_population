

import numpy as np


# ==============================================================
# B.1 – CODAGE ET DÉCODAGE
# ==============================================================

N_BITS = 10          # bits par chromosome
X_MIN = -5.0
X_MAX = 5.0
RANGE = X_MAX - X_MIN   # 10.0


def decode(chromosome):

    n = len(chromosome)
    decimal = sum(chromosome[i] * (2 ** (n - 1 - i)) for i in range(n))
    x = X_MIN + decimal * RANGE / (2 ** n - 1)
    return x


def encode(x):
    """Encode un réel x en binaire (pour illustration)."""
    decimal = round((x - X_MIN) * (2 ** N_BITS - 1) / RANGE)
    decimal = max(0, min(2 ** N_BITS - 1, decimal))
    bits = [(decimal >> (N_BITS - 1 - i)) & 1 for i in range(N_BITS)]
    return bits


def fitness(chromosome):
    """Fitness = f(x) = sin(x)·exp(sin(x)), à maximiser."""
    x = decode(chromosome)
    val = np.sin(x) * np.exp(np.sin(x))
    return float(val)


def precision():
    """Précision du codage."""
    return RANGE / (2 ** N_BITS - 1)


def show_coding_examples():
    """Illustre le codage sur 2 chromosomes choisis."""
    examples = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 0 → x_min
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],   # 1023 → x_max
        [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],   # 341 → intermédiaire
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],   # 512 → proche min à 0
    ]
    print("\nB.1 – Exemples de codage :")
    print(f"  {'Chromosome':<35} | {'Décimal':>8} | {'x':>8} | {'f(x)':>10}")
    print("  " + "-" * 70)
    for c in examples:
        d = int(''.join(map(str, c)), 2)
        x = decode(c)
        f_val = fitness(c)
        print(f"  {''.join(map(str,c)):<35} | {d:>8} | {x:>8.4f} | {f_val:>10.6f}")
    print(f"\n  Précision du codage : Δx = {precision():.6f}")


# ==============================================================
# B.2 – OPÉRATEURS GÉNÉTIQUES
# ==============================================================

def init_population(pop_size, rng):
    """Initialise une population aléatoire."""
    return [list(rng.integers(0, 2, N_BITS)) for _ in range(pop_size)]


# --- Sélection ---

def roulette_selection(population, fitnesses, rng):
    """
    Sélection par roulette (proportionnelle à la fitness).
    Décale les fitness pour garantir des valeurs positives.
    """
    f_arr = np.array(fitnesses)
    f_arr = f_arr - f_arr.min() + 1e-6   # décalage → toutes positives
    total = f_arr.sum()
    probs = f_arr / total
    idx = rng.choice(len(population), size=len(population), p=probs)
    return [population[i].copy() for i in idx]


def tournament_selection(population, fitnesses, rng, k=3):
    """
    Sélection par tournoi de taille k.
    Pour chaque individu à sélectionner, on choisit k candidats aléatoires
    et on prend le meilleur.
    """
    selected = []
    n = len(population)
    for _ in range(n):
        candidates = rng.choice(n, size=k, replace=False)
        best_idx = candidates[int(np.argmax([fitnesses[c] for c in candidates]))]
        selected.append(population[best_idx].copy())
    return selected


# --- Croisement bipoints ---

def two_point_crossover(parent1, parent2, pc, rng):
    """
    Croisement bipoints avec probabilité pc.
    Deux points de coupure aléatoires p1 < p2.
    """
    if rng.random() < pc:
        p1, p2 = sorted(rng.choice(N_BITS - 1, size=2, replace=False) + 1)
        child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
        child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]
        return child1, child2
    return parent1.copy(), parent2.copy()


# --- Mutation bit à bit ---

def bit_mutation(chromosome, pm, rng):
    """Chaque bit est inversé avec probabilité pm."""
    return [1 - b if rng.random() < pm else b for b in chromosome]


# ==============================================================
# ALGORITHME GÉNÉTIQUE COMPLET
# ==============================================================

def genetic_algorithm(pop_size=50, n_gen=100, pc=0.8, pm=0.01,
                       selection='roulette', seed=42):
    """
    AG complet.

    Retourne :
        best_x      : meilleur x trouvé
        best_f      : meilleure fitness
        best_hist   : historique de la meilleure fitness par génération
        mean_hist   : historique de la fitness moyenne par génération
        diversity   : diversité (écart-type de la fitness) par génération
    """
    rng = np.random.default_rng(seed)
    population = init_population(pop_size, rng)

    best_hist = []
    mean_hist = []
    diversity_hist = []
    global_best = None
    global_best_f = -np.inf

    for gen in range(n_gen):
        fitnesses = [fitness(ind) for ind in population]
        f_arr = np.array(fitnesses)

        best_idx = int(np.argmax(f_arr))
        if f_arr[best_idx] > global_best_f:
            global_best_f = f_arr[best_idx]
            global_best = population[best_idx].copy()

        best_hist.append(global_best_f)
        mean_hist.append(float(f_arr.mean()))
        diversity_hist.append(float(f_arr.std()))

        # Sélection
        if selection == 'roulette':
            selected = roulette_selection(population, fitnesses, rng)
        else:
            selected = tournament_selection(population, fitnesses, rng, k=3)

        # Croisement et mutation
        new_population = []
        for i in range(0, pop_size - 1, 2):
            c1, c2 = two_point_crossover(selected[i], selected[i + 1], pc, rng)
            new_population.append(bit_mutation(c1, pm, rng))
            new_population.append(bit_mutation(c2, pm, rng))

        # Élitisme : garder le meilleur de la génération précédente
        if len(new_population) < pop_size:
            new_population.append(global_best.copy())
        else:
            new_population[0] = global_best.copy()

        population = new_population[:pop_size]

    best_x = decode(global_best)
    return best_x, global_best_f, best_hist, mean_hist, diversity_hist


def run_ga_experiments():
    """Compare plusieurs configurations de l'AG."""
    configs = [
        # (pop_size, n_gen, pc, pm, selection, label)
        (20,  100, 0.8, 0.01, 'roulette',    'pop=20,  pm=0.01, roulette'),
        (50,  100, 0.8, 0.01, 'roulette',    'pop=50,  pm=0.01, roulette'),
        (100, 100, 0.8, 0.01, 'roulette',    'pop=100, pm=0.01, roulette'),
        (50,  100, 0.8, 0.01, 'tournament',  'pop=50,  pm=0.01, tournoi'),
        (50,  100, 0.8, 0.05, 'roulette',    'pop=50,  pm=0.05, roulette'),
        (50,  100, 0.8, 0.10, 'roulette',    'pop=50,  pm=0.10, roulette'),
        (50,  100, 0.6, 0.01, 'roulette',    'pop=50,  pc=0.60, roulette'),
        (50,  100, 0.9, 0.01, 'roulette',    'pop=50,  pc=0.90, roulette'),
    ]

    results = {}
    for pop_size, n_gen, pc, pm, sel, label in configs:
        best_x, best_f, bh, mh, dh = genetic_algorithm(
            pop_size=pop_size, n_gen=n_gen, pc=pc, pm=pm,
            selection=sel, seed=42)
        results[label] = {
            'best_x': best_x, 'best_f': best_f,
            'best_hist': bh, 'mean_hist': mh, 'div_hist': dh,
            'pop_size': pop_size, 'n_gen': n_gen, 'pc': pc, 'pm': pm,
        }
    return results


def print_ga_report(results):
    """Affiche le rapport de l'AG."""
    print("\n" + "=" * 70)
    print("B.2 – ALGORITHME GÉNÉTIQUE")
    print("=" * 70)

    # Vrai maximum par calcul analytique
    xs = np.linspace(-5, 5, 10000)
    true_max = float(np.max(np.sin(xs) * np.exp(np.sin(xs))))
    true_x = float(xs[np.argmax(np.sin(xs) * np.exp(np.sin(xs)))])
    print(f"\nMaximum analytique de f(x) : f({true_x:.4f}) = {true_max:.6f}")

    print(f"\n{'Configuration':<40} | {'x trouvé':>9} | {'f(x)':>10} | {'Erreur':>10}")
    print("-" * 80)
    for label, res in results.items():
        err = abs(res['best_f'] - true_max)
        print(f"{label:<40} | {res['best_x']:>9.4f} | {res['best_f']:>10.6f} | {err:>10.6f}")


# ==============================================================
# B.3 – ANALYSE DES SCHÈMES
# ==============================================================

def schema_order(schema):
    """Ordre d'un schème = nombre de bits définis (non '*')."""
    return sum(1 for b in schema if b != '*')


def schema_length(schema):
    """
    Longueur utile d'un schème = distance entre premier et dernier bit défini.
    u(H) = position_dernier - position_premier
    """
    positions = [i for i, b in enumerate(schema) if b != '*']
    if len(positions) <= 1:
        return 0
    return positions[-1] - positions[0]


def schema_destruction_prob(schema, pc, pm):
    """
    Probabilité estimée de destruction d'un schème.
    P_destroy ≈ 1 - (1 - pc · u(H)/(N-1)) · (1 - pm)^o(H)
    """
    o = schema_order(schema)
    u = schema_length(schema)
    n = len(schema)
    p_cross = 1 - pc * u / (n - 1) if n > 1 else 0
    p_mut = (1 - pm) ** o
    return 1 - p_cross * p_mut


def schema_count_in_population(population, schema):
    """Compte combien d'individus de la population correspondent au schème."""
    count = 0
    for ind in population:
        match = all(
            schema[i] == '*' or ind[i] == schema[i]
            for i in range(len(schema))
        )
        if match:
            count += 1
    return count


def analyze_schemas(pc=0.8, pm=0.01, pop_size=50, seed=42):
    """
    Analyse de 4 schèmes représentatifs.
    Schème H = séquence de {0, 1, '*'}
    """
    schemas = [
        # (schème, label, interpretation)
        (['1', '*', '*', '*', '*', '*', '*', '*', '*', '*'],
         "H1 = 1*********",
         "bit0=1 → x dans [0, 5] ; ordre 1, longueur 0"),

        (['1', '1', '*', '*', '*', '*', '*', '*', '*', '*'],
         "H2 = 11********",
         "2 premiers bits=1 → x > 2.5 ; ordre 2, longueur 1"),

        (['1', '0', '0', '0', '*', '*', '*', '*', '1', '*'],
         "H3 = 1000****1*",
         "bits précis, long schème ; ordre 4, longueur 8"),

        (['*', '*', '*', '*', '1', '1', '1', '1', '*', '*'],
         "H4 = ****1111**",
         "bloc central, ordre 4, longueur 3"),
    ]

    print("\n" + "=" * 70)
    print("B.3 – ANALYSE DES SCHÈMES")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    population = init_population(pop_size, rng)
    initial_fitnesses = [fitness(ind) for ind in population]

    print(f"\n{'Schème':<22} | {'o(H)':>5} | {'u(H)':>5} | {'P(dest)':>9} | "
          f"{'m(H,0)':>7} | Interprétation")
    print("-" * 90)

    for schema, label, interp in schemas:
        schema_int = [int(b) if b != '*' else '*' for b in schema]
        o = schema_order(schema_int)
        u = schema_length(schema_int)
        p_dest = schema_destruction_prob(schema_int, pc, pm)
        m0 = schema_count_in_population(population, schema_int)
        print(f"{label:<22} | {o:>5} | {u:>5} | {p_dest:>9.4f} | {m0:>7} | {interp}")

    print("\n--- Évolution des schèmes au fil des générations ---")
    # On suit l'évolution de H1 (simple) et H3 (complexe) sur 100 générations
    pop = init_population(pop_size, rng)
    schema_H1 = [1, '*', '*', '*', '*', '*', '*', '*', '*', '*']
    schema_H3 = [1, 0, 0, 0, '*', '*', '*', '*', 1, '*']

    print(f"  {'Gen':>4} | {'|H1|':>6} | {'|H3|':>6} | {'fit_moy':>10}")
    for gen in range(0, 100, 10):
        # Faire évoluer la population de 10 générations
        for _ in range(10):
            fitnesses_g = [fitness(ind) for ind in pop]
            selected = roulette_selection(pop, fitnesses_g,
                                          np.random.default_rng(gen))
            new_pop = []
            for i in range(0, pop_size - 1, 2):
                c1, c2 = two_point_crossover(selected[i], selected[i+1],
                                              pc, np.random.default_rng(gen+i))
                new_pop.append(bit_mutation(c1, pm, np.random.default_rng(gen+i)))
                new_pop.append(bit_mutation(c2, pm, np.random.default_rng(gen+i+1)))
            pop = new_pop[:pop_size]

        fitnesses_g = [fitness(ind) for ind in pop]
        c_H1 = schema_count_in_population(pop, schema_H1)
        c_H3 = schema_count_in_population(pop, schema_H3)
        print(f"  {gen+10:>4} | {c_H1:>6} | {c_H3:>6} | {np.mean(fitnesses_g):>10.4f}")

    print("\n--- Interprétation ---")
    print("• H1 (ordre 1, u=0) : très robuste, peu détruit → propagé rapidement")
    print("• H2 (ordre 2, u=1) : encore robuste, converge vite si fitness élevée")
    print("• H3 (ordre 4, u=8) : probabilité de destruction élevée → se propage")
    print("  difficilement malgré une bonne fitness → confirmation du théorème")
    print("  des schèmes de Holland")
    print("• H4 (bloc, u=3)    : intermédiaire, survit si fitness compétitive")


# ------------------------------------------------------------------
if __name__ == '__main__':
    show_coding_examples()

    print("\n=== Expériences AG ===")
    results = run_ga_experiments()
    print_ga_report(results)

    analyze_schemas()
