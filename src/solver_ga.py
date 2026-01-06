import numpy as np
from src.solver_base import Solver


class GA(Solver):
    """
    Genetic Algorithm for 3D-BPP
    - Real-coded GA (Random-key encoding)
    - Tournament selection
    - SBX crossover
    - Polynomial mutation
    - Elitism survival
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Population
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fitnesses = np.full(self.pop_size, np.inf)

        # GA parameters (tuned for 3D-BPP)
        self.pc = 0.9
        self.pm = 0.1          # IMPORTANT: higher mutation
        self.eta_c = 15
        self.eta_m = 20
        self.tournament_k = 3  # stronger selection

    def run(self):
        # Initial evaluation
        for i in range(self.pop_size):
            self.fitnesses[i] = self.evaluate(self.population[i])

        best_idx = np.argmin(self.fitnesses)
        self.best_solution = self.population[best_idx].copy()
        best_f = self.fitnesses[best_idx]
        self.history.append(best_f)

        for g in range(self.generations):
            offspring = []

            # ---- ELITISM ----
            elite = self.population[best_idx].copy()

            # ---- Reproduction ----
            while len(offspring) < self.pop_size:
                p1 = self.tournament()
                p2 = self.tournament()

                if np.random.rand() < self.pc:
                    c1, c2 = self.sbx_crossover(p1, p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                self.polynomial_mutation(c1)
                self.polynomial_mutation(c2)

                offspring.append(c1)
                if len(offspring) < self.pop_size:
                    offspring.append(c2)

            # Replace population
            self.population = np.array(offspring)

            # Insert elite
            self.population[0] = elite

            # Evaluate
            for i in range(self.pop_size):
                self.fitnesses[i] = self.evaluate(self.population[i])

            best_idx = np.argmin(self.fitnesses)
            best_f = self.fitnesses[best_idx]

            if best_f < min(self.history):
                self.best_solution = self.population[best_idx].copy()

            self.history.append(best_f)

            if (g + 1) % 10 == 0:
                print(
                    f"[GA] Gen {g+1} | "
                    f"Best = {best_f:.4f} | "
                    f"Avg = {np.mean(self.fitnesses):.4f} | "
                    f"Std = {np.std(self.fitnesses):.4f}"
                )

        return self.history
    def tournament(self):
        idxs = np.random.randint(0, self.pop_size, self.tournament_k)
        best = idxs[0]
        for i in idxs[1:]:
            if self.fitnesses[i] < self.fitnesses[best]:
                best = i
        return self.population[best]
    def sbx_crossover(self, p1, p2):
        rand = np.random.rand(self.dim)
        beta = np.empty(self.dim)

        mask = rand <= 0.5
        beta[mask] = (2.0 * rand[mask]) ** (1.0 / (self.eta_c + 1))
        beta[~mask] = (1.0 / (2.0 * (1.0 - rand[~mask]))) ** (1.0 / (self.eta_c + 1))

        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)

        return np.clip(c1, 0, 1), np.clip(c2, 0, 1)

    def polynomial_mutation(self, ind):
        rand = np.random.rand(self.dim)
        mask = rand < self.pm
        if not np.any(mask):
            return

        u = np.random.rand(np.sum(mask))
        delta = np.zeros_like(u)

        m1 = u < 0.5
        delta[m1] = (2.0 * u[m1]) ** (1.0 / (self.eta_m + 1)) - 1.0
        delta[~m1] = 1.0 - (2.0 * (1.0 - u[~m1])) ** (1.0 / (self.eta_m + 1))

        ind[mask] += delta
        np.clip(ind, 0, 1, out=ind)
