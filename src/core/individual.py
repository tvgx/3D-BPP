import numpy as np

class Individual:
    """
    Represents an individual solution in the evolutionary algorithm.
    Uses Random Key Encoding.
    Mỗi hộp i được đại diện bởi 3 gene thực trong đoạn [0, 1].
    """
    def __init__(self, chromosome: np.ndarray):
        """
        Initialize an individual.
        
        Args:
            chromosome: A numpy array representing the random keys.
                        Size should be 3 * number_of_items.
        """
        self.chromosome = chromosome
        self.fitness = float('inf')  # Adjusted Number of Bins (aNB) or similar.
        self.solution = None         # Decode result (Phenotype) - List of Bins
        
        # --- MFEA Attributes (Multitask Optimization) ---
        self.skill_factor = None     # The task ID (0..K-1) this individual is most skilled at.
        self.factorial_rank = {}     # Ranks in each task {task_id: rank}
            # lower rank is better (1 is best)
        self.scalar_fitness = 0.0    # 1.0 / factorial_rank[skill_factor]
        self.factorial_cost = {}     # Raw fitness/cost for each task {task_id: cost}

    def copy(self):
        """Creates a deep copy of the individual."""
        ind = Individual(self.chromosome.copy())
        ind.fitness = self.fitness
        # Solution might be complex object, usually re-decoded if needed, or referenced if read-only
        ind.solution = self.solution 
        ind.skill_factor = self.skill_factor
        ind.factorial_rank = self.factorial_rank.copy()
        ind.scalar_fitness = self.scalar_fitness
        ind.factorial_cost = self.factorial_cost.copy()
        return ind

    def __repr__(self):
        return f"<Individual fitness={self.fitness:.4f} skill_factor={self.skill_factor} scalar_fitness={self.scalar_fitness:.4f}>"
