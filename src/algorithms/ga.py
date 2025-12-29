import numpy as np
from src.algorithms.base_ec import BaseEC
from src.core.population import Population, Individual

class GA(BaseEC):
    def __init__(self, config, decoder):
        super().__init__(config)
        self.decoder = decoder
        self.items_count = len(decoder.items)
        
        # Hyperparameters
        params = config.get('parameters', {})
        self.pop_size_multiplier = params.get('population_size_multiplier', 15)
        self.pop_size = self.pop_size_multiplier * self.items_count
        
        # Ensure even pop_size
        if self.pop_size % 2 != 0: self.pop_size += 1
        
        self.crossover_prob = params.get('crossover_prob', 0.9)
        self.mutation_prob_factor = params.get('mutation_prob_factor', 1.0)
        self.dim = 3 * self.items_count
        self.mutation_prob = (1.0 / self.dim) * self.mutation_prob_factor
        
        self.population = Population(self.pop_size, self.items_count)
        
    def initialize(self):
        # Population already initialized in __init__
        # Just evaluate initial population
        self.evaluate_population(self.population.individuals)
        best = self.population.get_best()
        self.best_solution = best.chromosome.copy()
        self.best_fitness = best.fitness

    def evolve(self):
        generations = self.config.get('generations', 50)
        for gen in range(generations):
            self.step()
            print(f"Gen {gen}: Best Fitness (aNB) = {self.best_fitness:.4f}")

    def step(self):
        # 1. Selection and Reproduction
        offspring_pop = []
        
        # Elitism (Keep best)
        best_ind = self.population.get_best()
        # Ensure we pass the chromosome copy so we don't modify the elite by reference later if not careful
        best_clone = Individual(best_ind.chromosome.copy())
        best_clone.fitness = best_ind.fitness
        offspring_pop.append(best_clone)
        
        while len(offspring_pop) < self.pop_size:
            # Select 2 parents
            p1 = self.tournament_selection()
            p2 = self.tournament_selection()
            
            # Crossover
            c1_chrom, c2_chrom = self.crossover(p1.chromosome, p2.chromosome)
            
            # Mutation
            self.mutate(c1_chrom)
            self.mutate(c2_chrom)
            
            offspring_pop.append(Individual(c1_chrom))
            if len(offspring_pop) < self.pop_size:
                offspring_pop.append(Individual(c2_chrom))
                
        # Update population
        self.population.individuals = offspring_pop
        
        # Evaluate
        self.evaluate_population(self.population.individuals)
        
        # Update Global Best
        current_best = self.population.get_best()
        if current_best.fitness < self.best_fitness:
            self.best_fitness = current_best.fitness
            self.best_solution = current_best.chromosome.copy()

    def evaluate_population(self, individuals):
        for ind in individuals:
            # Assuming minimization
            if ind.fitness == float('inf'):
                ind.fitness = self.decoder.evaluate(ind)

    def tournament_selection(self, k=3):
        # Random selection of k individuals
        indices = np.random.randint(0, self.pop_size, k)
        candidates = [self.population.individuals[i] for i in indices]
        return min(candidates, key=lambda x: x.fitness)

    def crossover(self, p1, p2):
        if np.random.rand() < self.crossover_prob:
            # Uniform Crossover
            mask = np.random.rand(self.dim) < 0.5
            c1 = np.where(mask, p1, p2)
            c2 = np.where(mask, p2, p1)
            return c1, c2
        else:
            return p1.copy(), p2.copy()

    def mutate(self, chrom):
        # Gaussian mutation + Clip
        mask = np.random.rand(self.dim) < self.mutation_prob
        if np.any(mask):
            noise = np.random.normal(0, 0.1, np.sum(mask)) # sigma=0.1
            chrom[mask] += noise
            np.clip(chrom, 0, 1, out=chrom)
