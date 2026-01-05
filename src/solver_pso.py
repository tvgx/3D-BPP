import numpy as np
from src.solver_base import Solver

class PSO(Solver):
    """
    Particle Swarm Optimization (PSO).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize particles
        self.position = np.random.rand(self.pop_size, self.dim)
        self.velocity = np.random.uniform(-0.5, 0.5, (self.pop_size, self.dim))
        
        # Personal best
        self.pbest = self.position.copy()
        self.pbest_fitness = np.array([float('inf')] * self.pop_size)
        
        # Global best
        self.gbest = None
        self.gbest_fitness = float('inf')
        
        # PSO parameters
        self.w = 0.7          # Inertia weight
        self.c1 = 2.0         # Cognitive parameter
        self.c2 = 2.0         # Social parameter
        
    def run(self):
        # Evaluate initial population
        for i in range(self.pop_size):
            fitness = self.evaluate(self.position[i])
            self.pbest_fitness[i] = fitness
            
            if fitness < self.gbest_fitness:
                self.gbest_fitness = fitness
                self.gbest = self.position[i].copy()
        
        self.best_solution = self.gbest.copy()
        self.history.append(self.gbest_fitness)
        
        for g in range(self.generations):
            # Update velocity and position
            for i in range(self.pop_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                
                # Velocity update equation
                self.velocity[i] = (
                    self.w * self.velocity[i] +
                    self.c1 * r1 * (self.pbest[i] - self.position[i]) +
                    self.c2 * r2 * (self.gbest - self.position[i])
                )
                
                # Position update
                self.position[i] += self.velocity[i]
                self.position[i] = np.clip(self.position[i], 0, 1)
                
                # Evaluate new position
                fitness = self.evaluate(self.position[i])
                
                # Update personal best
                if fitness < self.pbest_fitness[i]:
                    self.pbest_fitness[i] = fitness
                    self.pbest[i] = self.position[i].copy()
                    
                    # Update global best
                    if fitness < self.gbest_fitness:
                        self.gbest_fitness = fitness
                        self.gbest = self.position[i].copy()
                        self.best_solution = self.gbest.copy()
            
            self.history.append(self.gbest_fitness)
            
            if (g+1) % 10 == 0:
                print(f"[PSO] Gen {g+1}: Best Fitness = {self.gbest_fitness:.4f}")
        
        return self.history
