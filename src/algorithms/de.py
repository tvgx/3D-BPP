import numpy as np
from src.algorithms.base_ec import BaseEC
from src.core.population import Individual, Population

class DE_SHADE(BaseEC):
    def __init__(self, config, decoder):
        super().__init__(config)
        self.decoder = decoder
        self.items_count = len(decoder.items)
        self.dim = 3 * self.items_count # 3n genes: sequence, rotation, heuristic
        
        self.pop_size = config.get('pop_size', 100)
        self.population = Population(self.pop_size, self.items_count)
        
        # Cấu hình SHADE
        self.memory_size = config.get('memory_size', 10)
        self.m_cr = np.full(self.memory_size, 0.5)
        self.m_f = np.full(self.memory_size, 0.5)
        self.k_idx = 0
        self.archive = []
        self.archive_limit = self.pop_size

    def initialize(self):
        # Population initialized in __init__
        pass

    def evolve(self):
        generations = self.config.get('generations', 100)
        for gen in range(generations):
            self.step()
            print(f"Gen {gen}: Best Fitness (aNB) = {self.best_fitness:.4f}")


    def step(self):
        pop = self.population.individuals
        scr, sf, diff_fits = [], [], []
        
        # Tạo tham số F và Cr cho từng cá thể qua phân phối Cauchy và Normal
        for i in range(self.pop_size):
            r = np.random.randint(0, self.memory_size)
            cr = np.clip(np.random.normal(self.m_cr[r], 0.1), 0, 1)
            while True:
                f = self.m_f[r] + 0.1 * np.random.standard_cauchy()
                if f > 0: break
            f = min(f, 1.0)

            # Đột biến Current-to-pbest/1
            p_best_idx = np.random.randint(0, max(1, int(0.1 * self.pop_size)))
            x_pbest = pop[p_best_idx].chromosome
            
            idxs = [idx for idx in range(self.pop_size) if idx != i]
            r1 = pop[np.random.choice(idxs)].chromosome
            
            # Sử dụng archive để tăng độ đa dạng
            pool = [p.chromosome for p in pop] + self.archive
            r2 = pool[np.random.randint(0, len(pool))]
            
            mutant = pop[i].chromosome + f * (x_pbest - pop[i].chromosome) + f * (r1 - r2)
            
            # Lai ghép nhị phân (Binomial Crossover)
            mask = np.random.rand(self.dim) < cr
            mask[np.random.randint(0, self.dim)] = True
            trial_chrom = np.clip(np.where(mask, mutant, pop[i].chromosome), 0, 1)
            
            trial_fit = self.decoder.get_fitness(trial_chrom)
            
            if trial_fit <= pop[i].fitness:
                if trial_fit < pop[i].fitness:
                    scr.append(cr); sf.append(f)
                    diff_fits.append(pop[i].fitness - trial_fit)
                    self.archive.append(pop[i].chromosome.copy())
                    if len(self.archive) > self.archive_limit:
                        self.archive.pop(np.random.randint(0, len(self.archive)))
                
                pop[i].chromosome = trial_chrom
                pop[i].fitness = trial_fit

        # Cập nhật bộ nhớ tham số Lehmer Mean
        if scr:
            weights = np.array(diff_fits) / sum(diff_fits)
            self.m_cr[self.k_idx] = np.sum(weights * np.array(scr))
            self.m_f[self.k_idx] = np.sum(weights * (np.array(sf)**2)) / np.sum(weights * np.array(sf))
            self.k_idx = (self.k_idx + 1) % self.memory_size