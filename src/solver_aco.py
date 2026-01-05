import numpy as np
from src.solver_base import Solver

class ACO(Solver):
    """
    Ant Colony Optimization (ACO).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # ACO parameters
        self.n_ants = self.pop_size
        self.alpha = 1.0          # Pheromone importance
        self.beta = 2.0           # Heuristic importance
        self.rho = 0.1            # Evaporation rate
        self.Q = 1.0              # Pheromone deposit amount
        
        # Pheromone matrix: dim x dim
        # Represents pheromone on edges between genes
        self.pheromone = np.ones((self.dim, self.dim)) * 0.1
        
        # Best solution tracking
        self.best_path = None
        self.best_fitness = float('inf')
        
    def run(self):
        for g in range(self.generations):
            ant_paths = []
            ant_fitnesses = []
            
            # Ant construction phase
            for ant_idx in range(self.n_ants):
                path = self.construct_solution()
                ant_paths.append(path)
                
                fitness = self.evaluate(path)
                ant_fitnesses.append(fitness)
                
                # Update global best
                if fitness < self.best_fitness:
                    self.best_fitness = fitness
                    self.best_path = path.copy()
                    self.best_solution = path.copy()
            
            # Pheromone evaporation
            self.pheromone *= (1 - self.rho)
            
            # Pheromone deposit
            for ant_idx in range(self.n_ants):
                path = ant_paths[ant_idx]
                fitness = ant_fitnesses[ant_idx]
                
                # Deposit pheromone proportional to fitness quality
                delta_pheromone = self.Q / fitness if fitness > 0 else self.Q
                
                # Update pheromone along the path
                for i in range(self.dim - 1):
                    gene_i = int(path[i] * (self.dim - 1))
                    gene_j = int(path[i + 1] * (self.dim - 1))
                    gene_i = np.clip(gene_i, 0, self.dim - 1)
                    gene_j = np.clip(gene_j, 0, self.dim - 1)
                    self.pheromone[gene_i, gene_j] += delta_pheromone
            
            # Clip pheromone to avoid extreme values
            self.pheromone = np.clip(self.pheromone, 0.01, 10.0)
            
            self.history.append(self.best_fitness)
            
            if (g+1) % 10 == 0:
                print(f"[ACO] Gen {g+1}: Best Fitness = {self.best_fitness:.4f}")
        
        return self.history
    
    def construct_solution(self):
        """
        Construct a solution using probabilistic selection based on pheromone and heuristic
        """
        solution = np.zeros(self.dim)
        
        # Start from random position
        current = np.random.randint(0, self.dim)
        solution[0] = current / self.dim
        visited = [current]
        
        # Build solution step by step
        for step in range(1, self.dim):
            # Calculate transition probabilities
            probabilities = np.zeros(self.dim)
            
            for next_gene in range(self.dim):
                if next_gene not in visited:
                    # Pheromone component
                    pheromone_val = self.pheromone[current, next_gene] ** self.alpha
                    
                    # Heuristic component (inverse of distance/difference)
                    heuristic_val = (1.0 / (abs(next_gene - current) + 1)) ** self.beta
                    
                    probabilities[next_gene] = pheromone_val * heuristic_val
            
            # Normalize probabilities
            prob_sum = np.sum(probabilities)
            if prob_sum > 0:
                probabilities /= prob_sum
            else:
                # If all probabilities are 0, choose uniformly from unvisited
                unvisited = [g for g in range(self.dim) if g not in visited]
                if unvisited:
                    probabilities[unvisited] = 1.0 / len(unvisited)
                else:
                    break
            
            # Select next gene probabilistically
            next_gene = np.random.choice(range(self.dim), p=probabilities)
            solution[step] = next_gene / self.dim
            visited.append(next_gene)
            current = next_gene
        
        return np.clip(solution, 0, 1)
    
    @property
    def best_bins(self):
        if self.best_solution is None: 
            return []
        from src.decoder import HeuristicDecoder
        return HeuristicDecoder.decode(self.best_solution, self.items, self.bin_dims, self.max_weight)
