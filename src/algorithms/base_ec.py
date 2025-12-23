from abc import ABC, abstractmethod

class BaseEC(ABC):
    """
    Abstract Base Class for Evolutionary Algorithms.
    """
    def __init__(self, config):
        self.config = config
        self.population = None
        self.best_solution = None
        self.best_fitness = float('inf')

    @abstractmethod
    def initialize(self):
        """Initialize population."""
        pass

    @abstractmethod
    def evolve(self):
        """Run the evolution process."""
        pass

    @abstractmethod
    def step(self):
        """Run one generation step."""
        pass
