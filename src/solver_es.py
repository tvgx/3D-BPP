import numpy as np
from src.solver_base import Solver

class ES(Solver):
    """
    (Mu + Lambda) Evolution Strategy.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = 20
        self.lam = 140
        self.population = [np.random.rand(self.dim) for _ in range(self.mu)]
        self.fitnesses = [self.evaluate(ind) for ind in self.population]
        
        self.sigma = 0.1 
        
    def run(self):
        best_idx = np.argmin(self.fitnesses)
        self.best_solution = self.population[best_idx].copy()
        best_f = self.fitnesses[best_idx]
        self.history.append(best_f)
        
        for g in range(self.generations):
            offspring = []
            offspring_fits = []
            
            for _ in range(self.lam):
                parent = self.population[np.random.randint(0, self.mu)]
                child = parent + np.random.normal(0, self.sigma, self.dim)
                child = np.clip(child, 0, 1)
                
                fit = self.evaluate(child)
                offspring.append(child)
                offspring_fits.append(fit)
            
            pool = self.population + offspring
            pool_fits = self.fitnesses + offspring_fits
            
            sorted_indices = np.argsort(pool_fits)
            self.population = [pool[i] for i in sorted_indices[:self.mu]]
            self.fitnesses = [pool_fits[i] for i in sorted_indices[:self.mu]]
            
            best_f = self.fitnesses[0]
            if best_f < min(self.history):
                 self.best_solution = self.population[0].copy()
            self.history.append(best_f)
            
            if (g+1) % 10 == 0:
                print(f"[ES] Gen {g+1}: Best Fitness = {best_f:.4f}")
        
        return self.history
