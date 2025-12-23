import numpy as np
from src.algorithms.base_ec import BaseEC
from src.core.population import Population
from src.core.skill_factor import SkillFactorManager
# Assume we have tasks defined elsewhere or passed in

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
        
        # Unified search space dimension should be max of all tasks
        # For 3D-BPP, tasks might describe different datasets (bins/items).
        # We take the max item count * 3 as chromosome length.
        max_items = max([t.num_items for t in tasks])
        self.dim = max_items * 3
        
        self.pop_size = config.get('pop_size', 100)
        self.population = Population(self.pop_size, max_items) # Need to adjust Population to accept dim directly or max_items
        
    def initialize(self):
        self.population.initialize()
        self.evaluate_all()
        
    def evaluate_all(self):
        # In MFEA, not every individual is evaluated on every task immediately in some variants,
        # but standard MFEA evaluates factorial cost for all?
        # Usually: evaluate only on skill factor task? No, initially all?
        # Standard MFEA: logic is complex.
        # Simplified: Evaluate all on all tasks (expensive) or just assign random skill factors initially?
        # Let's assume full evaluation for initialization.
        
        task_fitnesses = []
        for k, task in enumerate(self.tasks):
            fits = []
            for ind in self.population.individuals:
                # Task specific evaluation
                # Note: chromosome mapping might be needed if tasks have different dims
                f = task.evaluate(ind)
                fits.append(f)
            task_fitnesses.append(fits)
            
        SkillFactorManager.assign_ranks_and_skill_factors(self.population.individuals, task_fitnesses)

    def evolve(self):
        generations = self.config.get('generations', 100)
        for g in range(generations):
            self.step()
            print(f"Gen {g}: Best Scalar Fitness = {max([ind.scalar_fitness for ind in self.population.individuals]):.4f}")

    def step(self):
        # 1. Selection: Current population is P(t)
        # We need to generate offspring population C(t) of size N
        offspring_pop = []
        
        # We will generate offspring pairs
        while len(offspring_pop) < self.pop_size:
            # Select two parents using Tournament Selection on Scalar Fitness
            p1 = self._tournament_selection()
            p2 = self._tournament_selection()
            
            c1_chrom = p1.chromosome.copy()
            c2_chrom = p2.chromosome.copy()
            
            # Assortative Mating
            # [cite: Reference MFEA paper]
            can_mate = (p1.skill_factor == p2.skill_factor) or (np.random.rand() < self.rmp)
            
            if can_mate:
                # Crossover (Simulated Binary Crossover - SBX or Uniform)
                # Here we use a simple Uniform Crossover for Random Keys
                c1_chrom, c2_chrom = self._uniform_crossover(p1.chromosome, p2.chromosome)
                
                # Assign Skill Factors (Imitation)
                # Offspring mimic parents' skill factors
                # Valid method: randomly choose one from parents
                # Or if same skill factor, inherit that.
                
                sf1 = p1.skill_factor if np.random.rand() < 0.5 else p2.skill_factor
                sf2 = p1.skill_factor if np.random.rand() < 0.5 else p2.skill_factor
                
                # Refinement (Mutation) can still happen after crossover in some impls, 
                # but standard MFEA typically does Crossover OR Mutation path for structure.
                # However, many implementations do Crossover AND Mutation.
                # Let's apply mutation with low probability.
                c1_chrom = self._polynomial_mutation(c1_chrom)
                c2_chrom = self._polynomial_mutation(c2_chrom)
                
            else:
                # Mutation only
                c1_chrom = self._polynomial_mutation(p1.chromosome)
                c2_chrom = self._polynomial_mutation(p2.chromosome)
                
                # Skill factor: offspring inherits from its mutant parent
                sf1 = p1.skill_factor
                sf2 = p2.skill_factor
            
            # Create offspring objects
            # Note: We create 'Individual' but we haven't evaluated them yet.
            # In MFEA, we only evaluate on the skill factor task!
            
            from src.core.individual import Individual
            child1 = Individual(c1_chrom)
            child1.skill_factor = sf1
            
            child2 = Individual(c2_chrom)
            child2.skill_factor = sf2
            
            offspring_pop.append(child1)
            offspring_pop.append(child2)
            
        # Truncate if odd number
        if len(offspring_pop) > self.pop_size:
            offspring_pop.pop()
            
        # 2. Evaluation
        # Evaluate each offspring ONLY on their assigned skill factor
        for ind in offspring_pop:
            task_idx = ind.skill_factor
            cost = self.tasks[task_idx].evaluate(ind)
            ind.factorial_cost[task_idx] = cost
            # For other tasks, cost is infinity
            for k in range(self.num_tasks):
                if k != task_idx:
                    ind.factorial_cost[k] = float('inf')
        
        # 3. Concatenate P(t) and C(t)
        total_pop = self.population.individuals + offspring_pop
        
        # 4. Update Fitness (Scalar Fitness update for standard MFEA involves considering ranks across whole pool)
        # We need to re-rank everything
        task_fitnesses = []
        for k in range(self.num_tasks):
            # Extract costs for task k
            fits = [ind.factorial_cost.get(k, float('inf')) for ind in total_pop]
            task_fitnesses.append(fits)
            
        SkillFactorManager.assign_ranks_and_skill_factors(total_pop, task_fitnesses)
        
        # 5. Environmental Selection (Top N based on scalar fitness)
        total_pop.sort(key=lambda x: x.scalar_fitness, reverse=True) # Higher scalar fitness is better
        self.population.individuals = total_pop[:self.pop_size]

    def _tournament_selection(self, k=2):
        indices = np.random.choice(len(self.population.individuals), k, replace=False)
        candidates = [self.population.individuals[i] for i in indices]
        # Return max scalar fitness
        return max(candidates, key=lambda x: x.scalar_fitness)

    def _uniform_crossover(self, p1, p2):
        mask = np.random.rand(self.dim) < 0.5
        c1 = np.where(mask, p1, p2)
        c2 = np.where(mask, p2, p1)
        return c1, c2
    
    def _polynomial_mutation(self, p, eta=20, prob=0.1):
        # Simplified Polynomial Mutation for Random Keys [0, 1]
        c = p.copy()
        mask = np.random.rand(self.dim) < prob
        
        # For selected genes apply mutation. 
        # Real implementation is complex, using Gaussian noise for simplicity here as it fits Random Keys well.
        noise = np.random.normal(0, 0.1, size=self.dim)
        c = np.where(mask, c + noise, c)
        c = np.clip(c, 0, 1)
        return c
