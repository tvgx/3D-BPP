import numpy as np

class SkillFactorManager:
    """
    Manages Skill Factors and Factorial Ranks for Multifactorial Evolutionary Algorithm (MFEA).
    """
    
    @staticmethod
    def assign_ranks_and_skill_factors(population, task_fitnesses):
        """
        Assigns factorial ranks and skill factors to the population.
        
        Args:
            population: List of Individual objects.
            task_fitnesses: List of lists/arrays, where task_fitnesses[k][i] is the fitness of individual i on task k.
                            Dimensions: [NumTasks, PopSize]
        """
        pop_size = len(population)
        num_tasks = len(task_fitnesses)
        
        # Reset properties
        for ind in population:
            ind.factorial_rank = {}
            ind.skill_factor = None
            ind.scalar_fitness = 0.0
            
        # For each task, sort and assign ranks
        for k in range(num_tasks):
            # Get indices sorted by fitness for task k (ascending)
            # We assume we have the fitness values corresponding to the current population order
            fitness_k = task_fitnesses[k]
            sorted_indices = np.argsort(fitness_k)
            
            for rank, idx in enumerate(sorted_indices):
                population[idx].factorial_rank[k] = rank + 1 # 1-based rank
                population[idx].factorial_cost[k] = fitness_k[idx]

        # Assign skill factor and scalar fitness
        for ind in population:
            best_rank = float('inf')
            best_task = -1
            
            for k in range(num_tasks):
                rank = ind.factorial_rank.get(k, float('inf'))
                if rank < best_rank:
                    best_rank = rank
                    best_task = k
                elif rank == best_rank:
                    # Tie-breaking (random or first)
                    pass
            
            ind.skill_factor = best_task
            ind.scalar_fitness = 1.0 / best_rank
