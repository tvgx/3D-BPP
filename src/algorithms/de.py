import numpy as np
from src.algorithms.base_ec import BaseEC
from src.core.population import Individual, Population

class DE_SHADE(BaseEC):
    """
    SHADE: Success-History Based Parameter Adaptation for Differential Evolution.
    Implements:
    - Current-to-pbest mutation
    - M_F and M_Cr history archives
    """
    def __init__(self, config, decoder):
        super().__init__(config)
        self.decoder = decoder
        self.items_count = len(decoder.items)
        self.dim = 3 * self.items_count
        
        self.pop_size = config.get('pop_size', 100)
        self.population = Population(self.pop_size, self.items_count, init_random=True)
        
        # SHADE Parameters
        self.memory_size = config.get('memory_size', 5) # H
        self.m_cr = np.ones(self.memory_size) * 0.5
        self.m_f = np.ones(self.memory_size) * 0.5
        self.k_idx = 0  # memory index
        self.archive = [] # External archive for pbest if needed, simple SHADE uses A
        self.archive_size_max = self.pop_size # |A| <= N
        
        self.p_best_rate = 0.1 # top 100p%
        
    def initialize(self):
        self.population.initialize()
        self.evaluate_all() # Initial evaluation
        
    def evaluate_all(self):
        for ind in self.population.individuals:
            ind.fitness = self.decoder.get_fitness(ind.chromosome)
        
        self.population.sort()
        self.best_solution = self.population.individuals[0].chromosome.copy()
        self.best_fitness = self.population.individuals[0].fitness

    def evolve(self):
        generations = self.config.get('generations', 100)
        for g in range(generations):
            self.step()
            print(f"Gen {g}: Best Fitness (aNB) = {self.best_fitness:.4f}")

    def step(self):
        pop = self.population.individuals
        new_pop = []
        
        scr = [] # Successful CRs
        sf = []  # Successful Fs
        diff_fitness = []
        
        # Generate CR and F for each individual
        # using Cauchy for F and Normal for CR based on memory
        
        cr_vals = []
        f_vals = []
        
        for i in range(self.pop_size):
            # Pick random memory index
            r_idx = np.random.randint(0, self.memory_size)
            mu_cr = self.m_cr[r_idx]
            mu_f = self.m_f[r_idx]
            
            # Generate CR
            # Normal(mu_cr, 0.1), clamped [0, 1]
            cr = np.random.normal(mu_cr, 0.1)
            cr = np.clip(cr, 0, 1)
            cr_vals.append(cr)
            
            # Generate F
            # Cauchy(mu_f, 0.1)
            # If F > 1 -> 1, if F <= 0 -> regenerate
            while True:
                f = mu_f + 0.1 * np.random.standard_cauchy()
                if f > 0:
                    break
            f = min(f, 1.0)
            f_vals.append(f)
            
        # Evolution loop
        for i in range(self.pop_size):
            x_i = pop[i].chromosome
            f = f_vals[i]
            cr = cr_vals[i]
            
            # Current-to-pbest/1
            # v = x_i + F(x_pbest - x_i) + F(x_r1 - x_r2)
            
            # Select pbest from top p%
            p_best_count = max(1, int(self.p_best_rate * self.pop_size))
            p_best_ind = pop[np.random.randint(0, p_best_count)]
            x_pbest = p_best_ind.chromosome
            
            # Select r1 from Pop (distinct from i)
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            r1_idx = np.random.choice(idxs)
            x_r1 = pop[r1_idx].chromosome
            
            # Select r2 from Pop U Archive (distinct from i, r1)
            # Unified pool
            pool_candidates = [p.chromosome for p in pop] + self.archive
            # Logic to pick r2 distinct is tricky with raw arrays in archive
            # Simplified: Pick from pool, ensure not same object ref or value match?
            # Standard DE just picks random index.
            r2_idx = np.random.randint(0, len(pool_candidates))
            # Just retry if same (simplified)
            while r2_idx == i or r2_idx == r1_idx:
                 r2_idx = np.random.randint(0, len(pool_candidates))
            x_r2 = pool_candidates[r2_idx]
            
            # Mutation
            mutant = x_i + f * (x_pbest - x_i) + f * (x_r1 - x_r2)
            
            # Crossover (Binomial)
            j_rand = np.random.randint(0, self.dim)
            mask = np.random.rand(self.dim) < cr
            mask[j_rand] = True
            
            trial_chrom = np.where(mask, mutant, x_i)
            # Boundary handling (Random keys always [0, 1])
            # If violated, valid strategy: midpoint or random.
            # SHADE often uses: if > 1 -> (max+old)/2
            trial_chrom = np.clip(trial_chrom, 0, 1) # Simple clipping
            
            # Evaluation
            trial_ind = Individual(trial_chrom)
            trial_ind.fitness = self.decoder.get_fitness(trial_chrom)
            
            # Selection
            if trial_ind.fitness <= pop[i].fitness: # Better or equal
                new_pop.append(trial_ind)
                
                if trial_ind.fitness < pop[i].fitness:
                    scr.append(cr)
                    sf.append(f)
                    diff_fitness.append(pop[i].fitness - trial_ind.fitness)
                    
                    # Update Fitness Stats
                    if trial_ind.fitness < self.best_fitness:
                        self.best_fitness = trial_ind.fitness
                        self.best_solution = trial_ind.chromosome.copy()

                    # Add parent to archive
                    self.archive.append(x_i.copy())
                    if len(self.archive) > self.archive_size_max:
                        # Remove random
                        self.archive.pop(np.random.randint(0, len(self.archive)))
            else:
                new_pop.append(pop[i])
                
        # Update Population
        self.population.individuals = new_pop
        self.population.sort() # Keep sorted for p-best selection next gen
        
        # Update Memories if successes
        if scr:
            # Weighted Lehmer Mean for F (using diff_fitness as weights)
            weights = np.array(diff_fitness) / sum(diff_fitness)
            
            mean_scr = np.sum(weights * np.array(scr))
            mean_sf = np.sum(weights * (np.array(sf)**2)) / np.sum(weights * np.array(sf))
            
            self.m_cr[self.k_idx] = mean_scr
            self.m_f[self.k_idx] = mean_sf
            
            self.k_idx = (self.k_idx + 1) % self.memory_size