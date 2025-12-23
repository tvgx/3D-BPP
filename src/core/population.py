import numpy as np
from src.core.individual import Individual

class Population:
    """
    Manages a population of individuals.
    """
    def __init__(self, pop_size, num_items, init_random=True):
        self.pop_size = pop_size
        self.num_items = num_items
        self.individuals = []
        
        if init_random:
            self.initialize()

    def initialize(self):
        """Initializes the population with random individuals."""
        self.individuals = []
        for _ in range(self.pop_size):
            # 3 genes per item
            chromosome = np.random.rand(3 * self.num_items)
            self.individuals.append(Individual(chromosome))

    def get_best(self):
        """Returns the best individual in the population."""
        if not self.individuals:
            return None
        # Assuming lower fitness is better (minimization)
        return min(self.individuals, key=lambda x: x.fitness)

    def add(self, individual):
        self.individuals.append(individual)

    def sort(self):
        self.individuals.sort(key=lambda x: x.fitness)

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, idx):
        return self.individuals[idx]
