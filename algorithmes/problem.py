"""
problem.py – Définition du problème d'optimisation binaire

Fonction de coût :
    f(s) = Σ_i α_i·b_i  +  Σ_{i<j} β_ij·b_i·b_j

avec n = 10 bits, des coefficients choisis pour créer plusieurs minima locaux.

Justification des coefficients :
  - Les αᵢ alternent entre valeurs négatives (qui encouragent bᵢ=1) et positive
    (qui découragent bᵢ=1), créant des tensions sans interactions.
  - Les βᵢⱼ introduisent des pénalités ou récompenses entre paires de bits,
    ce qui génère des conflits locaux (minima locaux différents du minimum global).
"""

import numpy as np
import itertools


class BinaryProblem:
    """
    Problème de minimisation sur {0,1}^n.
    n = 10  (exigence : n >= 8)
    """

    def __init__(self):
        self.n = 10

        # ---- Coefficients linéaires α ----
        # Certains négatifs (→ voudrait bᵢ=1) et certains positifs (→ voudrait bᵢ=0)
        self.alpha = np.array([
            -3.0,  # b0 : fortement encouragé à 1
             2.0,  # b1 : découragé à 1
            -1.5,  # b2 : légèrement encouragé à 1
             4.0,  # b3 : fortement découragé à 1
            -2.5,  # b4 : encouragé à 1
             1.0,  # b5 : légèrement découragé à 1
            -3.5,  # b6 : fortement encouragé à 1
             2.5,  # b7 : découragé à 1
            -1.0,  # b8 : légèrement encouragé à 1
             3.0,  # b9 : découragé à 1
        ])

        # ---- Coefficients d'interaction β (matrice triangulaire supérieure) ----
        # Valeurs positives → pénalité si les deux bits sont à 1 (crée conflits)
        # Valeurs négatives → récompense si les deux bits sont à 1
        self.beta = np.zeros((self.n, self.n))

        interactions = [
            (0, 1,  5.0),   # pénalise fortement b0=b1=1  (conflit)
            (0, 4, -4.0),   # récompense b0=b4=1          (synergie)
            (1, 2,  3.0),   # pénalise b1=b2=1
            (2, 6, -3.5),   # récompense b2=b6=1
            (3, 7,  4.5),   # pénalise b3=b7=1
            (4, 5, -2.0),   # récompense b4=b5=1
            (5, 8,  3.0),   # pénalise b5=b8=1
            (6, 9, -4.0),   # récompense b6=b9=1
            (7, 8,  2.5),   # pénalise b7=b8=1
            (8, 9, -1.5),   # récompense b8=b9=1
            (0, 9,  2.0),   # pénalise b0=b9=1
            (1, 5, -1.5),   # récompense b1=b5=1
        ]
        for i, j, v in interactions:
            self.beta[i][j] = v

    # ------------------------------------------------------------------
    def cost(self, s):
        """Calcule f(s) pour un vecteur binaire s (list ou np.array)."""
        s = np.asarray(s, dtype=float)
        linear = float(self.alpha @ s)
        quad = 0.0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                quad += self.beta[i, j] * s[i] * s[j]
        return linear + quad

    def neighbors(self, s):
        """Retourne tous les voisins de Hamming-1 (flip d'un seul bit)."""
        s = list(s)
        result = []
        for i in range(self.n):
            nb = s.copy()
            nb[i] = 1 - nb[i]
            result.append(nb)
        return result

    def random_solution(self, rng=None):
        """Génère une solution binaire aléatoire."""
        if rng is None:
            return [int(b) for b in np.random.randint(0, 2, self.n)]
        return [int(b) for b in rng.integers(0, 2, self.n)]

    def solution_str(self, s):
        return ''.join(map(str, s))

    # ------------------------------------------------------------------
    def brute_force(self):
        """Force brute : énumère les 2^n = 1024 solutions (n=10 → faisable)."""
        best_cost = float('inf')
        best_sol = None
        local_minima = []

        for idx in range(2 ** self.n):
            s = [int(b) for b in format(idx, f'0{self.n}b')]
            c = self.cost(s)
            if c < best_cost:
                best_cost = c
                best_sol = s

        # Détection des minima locaux
        for idx in range(2 ** self.n):
            s = [int(b) for b in format(idx, f'0{self.n}b')]
            c = self.cost(s)
            is_local_min = all(self.cost(nb) >= c for nb in self.neighbors(s))
            if is_local_min:
                local_minima.append((s, c))

        local_minima.sort(key=lambda x: x[1])
        return best_sol, best_cost, local_minima


# ------------------------------------------------------------------
if __name__ == '__main__':
    prob = BinaryProblem()
    print("=== Problème d'optimisation binaire ===")
    print(f"n = {prob.n} bits  →  {2**prob.n} solutions possibles")
    print(f"\nCoefficients α : {prob.alpha}")
    print(f"\nMatrice β (interactions non nulles) :")
    for i in range(prob.n):
        for j in range(i + 1, prob.n):
            if prob.beta[i, j] != 0:
                sign = "pénalité" if prob.beta[i, j] > 0 else "récompense"
                print(f"  β[{i},{j}] = {prob.beta[i,j]:+.1f}  ({sign})")

    print("\nRecherche du minimum global par force brute...")
    best, best_c, local_mins = prob.brute_force()
    print(f"\nMinimum global : s* = {prob.solution_str(best)}  f(s*) = {best_c:.4f}")
    print(f"\nNombre de minima locaux : {len(local_mins)}")
    print("Minima locaux (10 premiers) :")
    for sol, c in local_mins[:10]:
        marker = " ← GLOBAL" if c == best_c else ""
        print(f"  {prob.solution_str(sol)}  f = {c:.4f}{marker}")
