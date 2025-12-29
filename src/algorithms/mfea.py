import numpy as np
from src.algorithms.base_ec import BaseEC
from src.core.population import Population
from src.core.skill_factor import SkillFactorManager
from src.core.individual import Individual

class MFEA(BaseEC):
    """
    Multifactorial Evolutionary Algorithm.
    """
    def __init__(self, config, tasks):
        """
        Args:
            config: Configuration dictionary.
            tasks: List of task objects (each with an evaluate function).
        """
        super().__init__(config)
        self.tasks = tasks
        self.num_tasks = len(tasks)
        self.rmp = config.get('rmp', 0.3) # Random Mating Probability
        
        # Unified search space dimension
        max_items = max([t.num_items for t in tasks])
        self.dim = max_items * 3
        
        self.pop_size = config.get('pop_size', 100)
        self.population = Population(self.pop_size, max_items)
        
    def initialize(self):
        self.population.initialize()
        self.evaluate_all()
        
    def evaluate_all(self):
        # Initial evaluation: evaluate on all tasks or assign random?
        # Standard MFEA: Calculate factorial costs for all tasks
        task_fitnesses = []
        for k, task in enumerate(self.tasks):
            fits = []
            for ind in self.population.individuals:
                # Note: Assuming chromosome length matches or is handled
                f = task.evaluate(ind)
                fits.append(f)
            task_fitnesses.append(fits)
            
        SkillFactorManager.assign_ranks_and_skill_factors(self.population.individuals, task_fitnesses)

    def evolve(self):
        generations = self.config.get('generations', 100)
        for g in range(generations):
            self.step()
            # Log progress
            best_ind = max(self.population.individuals, key=lambda x: x.scalar_fitness)
            print(f"Gen {g}: Best Scalar Fitness = {best_ind.scalar_fitness:.4f}")

    def step(self):
        offspring = []
        while len(offspring) < self.pop_size:
            # Selection
            p1 = self._tournament_selection(k=2)
            p2 = self._tournament_selection(k=2)
            
            # Assortative Mating
            if p1.skill_factor == p2.skill_factor or np.random.rand() < self.rmp:
                # Crossover
                c1, c2 = self._uniform_crossover(p1.chromosome, p2.chromosome)
                sf1 = p1.skill_factor if np.random.rand() < 0.5 else p2.skill_factor
                sf2 = p2.skill_factor if np.random.rand() < 0.5 else p1.skill_factor
            else:
                # Mutation
                c1, c2 = self._mutate(p1.chromosome), self._mutate(p2.chromosome)
                sf1, sf2 = p1.skill_factor, p2.skill_factor
            
            # Create offspring
            for c, sf in [(c1, sf1), (c2, sf2)]:
                child = Individual(c)
                child.skill_factor = sf
                # Only evaluate on assigned task
                cost = self.tasks[sf].evaluate(child)
                child.factorial_cost[sf] = cost
                # Set others to inf
                for k in range(self.num_tasks):
                    if k != sf:
                        child.factorial_cost[k] = float('inf')
                offspring.append(child)

        # Merge and Truncate
        total_pop = self.population.individuals + offspring
        
        # Update Ranks and Skill Factors (re-evaluate ranks based on costs)
        task_fitnesses = []
        for k in range(self.num_tasks):
            fits = [ind.factorial_cost.get(k, float('inf')) for ind in total_pop]
            task_fitnesses.append(fits)
        
        SkillFactorManager.assign_ranks_and_skill_factors(total_pop, task_fitnesses)
        
        # Environmental Selection (Scalar Fitness)
        total_pop.sort(key=lambda x: x.scalar_fitness, reverse=True)
        self.population.individuals = total_pop[:self.pop_size]

    def _tournament_selection(self, k=2):
        indices = np.random.choice(len(self.population.individuals), k, replace=False)
        candidates = [self.population.individuals[i] for i in indices]
        return max(candidates, key=lambda x: x.scalar_fitness)

    def _uniform_crossover(self, p1, p2):
        mask = np.random.rand(self.dim) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2
    
    def _mutate(self, p, prob=0.1):
        c = p.copy()
        mask = np.random.rand(self.dim) < prob
        noise = np.random.normal(0, 0.1, size=self.dim)
        c = np.where(mask, c + noise, c)
        c = np.clip(c, 0, 1)
        return c