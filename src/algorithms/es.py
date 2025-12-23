import numpy as np
from src.algorithms.base_ec import BaseEC

class CMA_ES(BaseEC):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    Implements covariance matrix C and evolution paths pc, ps updates.
    """
    def __init__(self, config, decoder):
        super().__init__(config)
        self.decoder = decoder
        self.items_count = len(decoder.items)
        self.dim = 3 * self.items_count
        
        # CMA-ES Parameters
        self.pop_size = config.get('pop_size', 4 + int(3 * np.log(self.dim))) # lambda
        self.mu = self.pop_size // 2
        
        # Weights (logarithmic)
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = 1 / np.sum(self.weights**2) # Variance effective selection mass
        
        # State Variables
        self.mean = np.random.rand(self.dim)
        self.sigma = 0.3 # Initial step size
        self.C = np.eye(self.dim) # Covariance Matrix
        self.pc = np.zeros(self.dim) # Evolution path for C
        self.ps = np.zeros(self.dim) # Evolution path for sigma
        self.B = np.eye(self.dim)
        self.D = np.ones(self.dim)
        
        # Adaptation constants
        self.cc = (4 + self.mueff/self.dim) / (self.dim + 4 + 2*self.mueff/self.dim)
        self.cs = (self.mueff + 2) / (self.dim + self.mueff + 5)
        self.c1 = 2 / ((self.dim + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((self.dim + 2)**2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.dim + 1)) - 1) + self.cs
        
        self.chiN = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        self.count_eigen = 0

    def initialize(self):
        # Already initialized in __init__ for CMA state
        pass

    def evolve(self):
        generations = self.config.get('generations', 100)
        for g in range(generations):
            self.step()
            print(f"Gen {g}: Best Fitness (aNB) = {self.best_fitness:.4f}")

    def step(self):
        # 1. Sampling
        offspring = []
        offspring_fits = []
        
        # Eigendecomposition C = B D^2 B^T
        # To sample: y ~ N(0, C) => y = B*D*z, z ~ N(0, I)
        # x = m + sigma * y
        
        # Recompute eigendecomposition if needed (O(N^3), lazy update)
        if self.count_eigen > (1.0 / (self.c1 + self.cmu) / self.dim / 10):
            self.count_eigen = 0
            self.C = np.triu(self.C) + np.triu(self.C, 1).T # Enforce symmetry
            d, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(d, 0)) # D contains std devs
            
        self.count_eigen += 1
            
        z_vals = []
        
        for k in range(self.pop_size):
            z = np.random.normal(0, 1, self.dim)
            z_vals.append(z)
            
            y = self.B @ (self.D * z)
            x = self.mean + self.sigma * y
            
            # Boundary handling for Random Keys [0, 1]
            # CMA-ES works in unconstrained usually. 
            # We treat x as genotype, but decode needs [0, 1].
            # We can use sigmoid wrapper or clipping.
            # Clipping is bad for CMA-ES distribution learning.
            # Mirroring or penalty is better.
            # Simple approach: Decode using clip, but keep x for update.
            
            x_decoded = np.clip(x, 0, 1) # Phenotype map
            
            fit = self.decoder.get_fitness(x_decoded)
            offspring.append(x)
            offspring_fits.append(fit)
            
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_solution = x_decoded.copy() # Store phenotype
                
        # 2. Selection and Recombination
        # Sort by fitness
        sort_indices = np.argsort(offspring_fits)
        best_indices = sort_indices[:self.mu]
        
        old_mean = self.mean.copy()
        
        # Compute new mean
        # m_new = sum(w_i * x_i)
        
        # Efficient way:
        # z_w = sum(w_i * z_i)
        # m_new = m_old + sigma * B * D * z_w
        
        term = np.zeros(self.dim)
        for i, idx in enumerate(best_indices):
            term += self.weights[i] * z_vals[idx]
        
        z_w = term
        
        # Update mean
        y_w = self.B @ (self.D * z_w)
        self.mean = old_mean + self.sigma * y_w
        
        # 3. Step-size control (Path for sigma)
        # ps = (1-cs)ps + sqrt(cs(2-cs)mueff) * B * z_w
        # Note: B * z_w = inv(D) * inv(B) * y_w ... effectively transforming back to isotropic
        
        inv_sqrt_act = np.sqrt(self.cs * (2 - self.cs) * self.mueff)
        self.ps = (1 - self.cs) * self.ps + inv_sqrt_act * (self.B @ z_w)
        
        norm_ps = np.linalg.norm(self.ps)
        
        # Update sigma
        self.sigma *= np.exp((self.cs / self.damps) * (norm_ps / self.chiN - 1))
        
        # 4. Covariance Matrix Adaptation (Path for C)
        # hsig check
        hsig = 1.0 # simplified logic, usually check norm_ps
        # if norm_ps / sqrt(1-(1-cs)^(2k)) < 1.4 + 2/(d+1) * chiN ...
        
        inv_sqrt_act_c = np.sqrt(self.cc * (2 - self.cc) * self.mueff)
        self.pc = (1 - self.cc) * self.pc + hsig * inv_sqrt_act_c * y_w
        
        # Rank-1 parameter
        c1a = self.c1 * (1 - (1-hsig**2)*self.cc*(2-self.cc)) 
        
        # Rank-mu update
        rank_mu = np.zeros((self.dim, self.dim))
        for i, idx in enumerate(best_indices):
            y_i = self.B @ (self.D * z_vals[idx]) # Reconstruct y
            # Actually y_i = (x_i - m_old)/sigma
            rank_mu += self.weights[i] * np.outer(y_i, y_i)
            
        self.C = (1 - c1a - self.cmu) * self.C \
                 + self.c1 * np.outer(self.pc, self.pc) \
                 + self.cmu * rank_mu
