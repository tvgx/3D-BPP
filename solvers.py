import numpy as np
import copy
from core import HeuristicDecoder

class Solver:
    def __init__(self, items, bin_dims, max_weight, pop_size, generations):
        self.items = items
        self.bin_dims = bin_dims
        self.max_weight = max_weight
        self.pop_size = pop_size
        self.generations = generations
        self.history = []
        self.dim = 2 * len(items) # Priority + Rotation keys
        self.best_solution = None
        
    @property
    def best_bins(self):
        if self.best_solution is None: return []
        return HeuristicDecoder.decode(self.best_solution, self.items, self.bin_dims, self.max_weight)

    def evaluate(self, chromosome):
        bins = HeuristicDecoder.decode(chromosome, self.items, self.bin_dims, self.max_weight)
        n_bins = len(bins)
        if n_bins == 0: return float('inf')
        
        total_vol = sum([i[0].dims[0]*i[0].dims[1]*i[0].dims[2] for b in bins for i in b.items])
        capacity = n_bins * (self.bin_dims[0] * self.bin_dims[1] * self.bin_dims[2])
        fill_rate = total_vol / capacity
        
        return n_bins + (1.0 - fill_rate)

class GA(Solver):
    """
    Genetic Algorithm with SBX Crossover and Polynomial Mutation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population = np.random.rand(self.pop_size, self.dim)
        self.fitnesses = np.array([float('inf')] * self.pop_size)
        
        # SBX/Poly params
        self.eta_c = 20
        self.eta_m = 20
        self.pc = 0.9
        self.pm = 1.0 / self.dim
        
    def run(self):
        # Eval initial
        for i in range(self.pop_size):
            self.fitnesses[i] = self.evaluate(self.population[i])
            
        best_idx = np.argmin(self.fitnesses)
        self.best_solution = self.population[best_idx].copy()
        self.history.append(self.fitnesses[best_idx])
        
        for g in range(self.generations):
            offspring = []
            
            # Tournament Selection & Crossover
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
                if len(offspring) < self.pop_size: offspring.append(c2)
            
            self.population = np.array(offspring)
            # Evaluate
            for i in range(self.pop_size):
                self.fitnesses[i] = self.evaluate(self.population[i])
                
            best_idx_gen = np.argmin(self.fitnesses)
            best_f = self.fitnesses[best_idx_gen]
            if best_f < min(self.history):
                 self.best_solution = self.population[best_idx_gen].copy()
            self.history.append(best_f)
            if (g+1) % 10 == 0:
                print(f"[GA] Gen {g+1}: Best Fitness = {best_f:.4f}")
                
        return self.history

    def tournament(self, k=2):
        idxs = np.random.randint(0, self.pop_size, k)
        best_i = idxs[0]
        for i in idxs[1:]:
            if self.fitnesses[i] < self.fitnesses[best_i]:
                best_i = i
        return self.population[best_i]

    def sbx_crossover(self, p1, p2):
        # Simulated Binary Crossover
        c1, c2 = p1.copy(), p2.copy()
        rand = np.random.rand(self.dim)
        beta = np.empty(self.dim)
        
        mask1 = rand <= 0.5
        beta[mask1] = (2.0 * rand[mask1])**(1.0/(self.eta_c+1))
        beta[~mask1] = (1.0 / (2.0 * (1.0 - rand[~mask1])))**(1.0/(self.eta_c+1))
        
        c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
        c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
        
        return np.clip(c1, 0, 1), np.clip(c2, 0, 1)

    def polynomial_mutation(self, ind):
        rand = np.random.rand(self.dim)
        mask = rand < self.pm
        if not np.any(mask): return
        
        u = np.random.rand(np.sum(mask))
        delta = np.zeros_like(u)
        m1 = u < 0.5
        delta[m1] = (2.0 * u[m1])**(1.0/(self.eta_m+1)) - 1.0
        delta[~m1] = 1.0 - (2.0 * (1.0 - u[~m1]))**(1.0/(self.eta_m+1))
        
        ind[mask] += delta
        np.clip(ind, 0, 1, out=ind)

class ES(Solver):
    """
    (Mu + Lambda) Evolution Strategy.
    Mu = 20, Lambda = 140.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mu = 20
        self.lam = 140
        # Initialize population of size Mu
        self.population = [np.random.rand(self.dim) for _ in range(self.mu)]
        self.fitnesses = [self.evaluate(ind) for ind in self.population]
        
        self.sigma = 0.1 # Mutation strength
        
    def run(self):
        best_idx = np.argmin(self.fitnesses)
        self.best_solution = self.population[best_idx].copy()
        best_f = self.fitnesses[best_idx]
        self.history.append(best_f)
        
        for g in range(self.generations):
            offspring = []
            offspring_fits = []
            
            # Generate Lambda offspring from Mu parents
            for _ in range(self.lam):
                # Random parent selection
                parent = self.population[np.random.randint(0, self.mu)]
                child = parent + np.random.normal(0, self.sigma, self.dim)
                child = np.clip(child, 0, 1)
                
                fit = self.evaluate(child)
                offspring.append(child)
                offspring_fits.append(fit)
            
            # (Mu + Lambda) Selection
            # Pool parents and offspring
            pool = self.population + offspring
            pool_fits = self.fitnesses + offspring_fits
            
            # Sort and select best Mu
            sorted_indices = np.argsort(pool_fits)
            self.population = [pool[i] for i in sorted_indices[:self.mu]]
            self.fitnesses = [pool_fits[i] for i in sorted_indices[:self.mu]]
            
            best_f = self.fitnesses[0]
            if best_f < min(self.history):
                 # Population was sorted, so index 0 is best
                 self.best_solution = self.population[0].copy()
            self.history.append(best_f)
            
            if (g+1) % 10 == 0:
                print(f"[ES] Gen {g+1}: Best Fitness = {best_f:.4f}")
        
        return self.history