import numpy as np
from src.algorithms.base_ec import BaseEC
from src.core.population import Individual, Population

class ACO(BaseEC):
    """
    Ant Colony Optimization for Permutation (Sequencing).
    Uses Pheromone Matrix tau[i][j] to represent probability of picking item j after item i.
    """
    def __init__(self, config, decoder):
        super().__init__(config)
        self.decoder = decoder
        self.items_count = len(decoder.items)
        
        self.pop_size = config.get('pop_size', 50) # Number of ants
        self.alpha = config.get('alpha', 1.0) # Pheromone weight
        self.beta = config.get('beta', 2.0)   # Heuristic weight
        self.rho = config.get('rho', 0.1)     # Evaporation rate
        self.q = config.get('q', 100.0)       # Pheromone deposit factor
        
        # Pheromone Matrix (N+1 x N) to account for 'start' node (virtual node N)
        # tau[i][j] means edge i -> j
        self.tau = np.ones((self.items_count + 1, self.items_count)) * 1.0
        
        # Heuristic Matrix (Static)
        # eta[i][j]: preference of j. Simple heuristic: Volume of j? 
        # For Bin Packing, bigger moves first is better (BFD). So Volume(j) is good heuristic.
        self.eta = np.ones(self.items_count)
        # Normalize volumes
        volumes = np.array([item.volume for item in decoder.items])
        self.eta = volumes / np.max(volumes)
        
        self.population = [] # List of solutions (sequences)

    def initialize(self):
        self.tau[:] = 1.0
        self.best_solution = None
        self.best_fitness = float('inf')

    def evolve(self):
        generations = self.config.get('generations', 100)
        for g in range(generations):
            self.step()
            print(f"Gen {g}: Best Fitness (aNB) = {self.best_fitness:.4f}")

    def step(self):
        # 1. Ant Construction
        solutions = []
        fitnesses = []
        
        for k in range(self.pop_size):
            # Construct a tour (sequence of items)
            visited = [False] * self.items_count
            sequence = []
            current_node = self.items_count # Start node index (virtual)
            
            for _ in range(self.items_count):
                # Calculate probabilities for unvisited nodes
                unvisited_idxs = [i for i in range(self.items_count) if not visited[i]]
                
                probs = []
                for j in unvisited_idxs:
                    # tau of edge (current, j) * eta of Node j
                    p = (self.tau[current_node][j] ** self.alpha) * (self.eta[j] ** self.beta)
                    probs.append(p)
                
                probs = np.array(probs)
                
                # Check for zero sum (rare but possible if pheromone decays too much)
                if np.sum(probs) == 0:
                    probs = np.ones(len(probs)) # Uniform backup
                    
                probs = probs / np.sum(probs)
                
                # Roulette Wheel Selection
                next_node_idx_in_list = np._default_rng().choice(len(unvisited_idxs), p=probs)
                next_node = unvisited_idxs[next_node_idx_in_list]
                
                sequence.append(next_node)
                visited[next_node] = True
                current_node = next_node
            
            # Create solution
            # Note: ACO generates Sequence. But we also need Rotation and Heuristic for full Genome.
            # Simplified Hybrid ACO: ACO optimizes Sequence. Rotation/Heuristic are random or evolved separately?
            # Or assume fixed/heuristic for now.
            # Let's generate a full random key based on the sequence.
            # Convert sequence to random keys: 
            # item at index i in sequence gets value i/N? No, rank.
            # If sequence is [2, 0, 1], then item 0 is at pos 1, item 1 at pos 2, item 2 at pos 0.
            # R1 for item 0 = 1/N, item 1 = 2/N, item 2 = 0/N.
            
            # Reconstruct Random Key Chromosome
            # Part 1: Order
            r1 = np.zeros(self.items_count)
            for rank, item_id in enumerate(sequence):
                r1[item_id] = rank / self.items_count
            
            # Part 2 & 3: Rotations and Heuristics
            # For pure ACO focused on Sequencing, we can randomize these or keep ‘best known’ 
            # Or just randomize for diversity.
            r_rest = np.random.rand(2 * self.items_count)
            
            chromosome = np.concatenate([r1, r_rest])
            
            # Evaluate
            fit = self.decoder.get_fitness(chromosome)
            solutions.append(sequence)
            fitnesses.append(fit)
            
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_solution = chromosome.copy()
        
        # 2. Pheromone Update
        # Evaporation
        self.tau *= (1 - self.rho)
        
        # Deposit (Global Best Update or Iteration Best?)
        # Let's use Iteration Best + Global Best to prevent stagnation
        best_idx = np.argmin(fitnesses)
        best_seq = solutions[best_idx]
        best_fit = fitnesses[best_idx]
        
        # Deposit amount
        delta_tau = self.q / best_fit # fitness is aNB (minimized)
        
        # Update for the best ant
        current_node = self.items_count
        for next_node in best_seq:
            self.tau[current_node][next_node] += delta_tau
            current_node = next_node
            
        # Optional: Min-Max pheromone limits (MMAS) to prevent stagnation
        self.tau = np.clip(self.tau, 0.001, 100.0)
