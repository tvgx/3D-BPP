import numpy as np
from src.algorithms.base_solver import BaseSolver

class BRKGA(BaseSolver):
    """
    Biased Random-Key Genetic Algorithm.
    Dựa trên tài liệu [cite: 1148] và[cite: 1207].
    """
    def initialize(self):
        # Population initialized in BaseSolver.__init__
        pass

    def evolve(self):
        params = self.config.get('brkga', {})
        num_elite = int(params.get('elite_pct', 0.2) * self.pop_size)
        num_mutant = int(params.get('mutant_pct', 0.2) * self.pop_size)
        num_crossover = self.pop_size - num_elite - num_mutant
        rho = params.get('inheritance_prob', 0.7) # Xác suất lấy gen từ Elite [cite: 1271]

        for gen in range(self.config.get('generations', 100)):
            self.step(num_elite, num_mutant, num_crossover, rho)
            print(f"Gen {gen}: Best Fitness aNB = {self.best_fitness:.4f}")

    def step(self, num_elite, num_mutant, num_crossover, rho):
        # 1. Đánh giá và Sắp xếp quần thể
        self.evaluate_all()
        sorted_indices = np.argsort(self.fitnesses)
        sorted_pop = self.population[sorted_indices]
        
        # 2. Phân loại Elite và Non-Elite [cite: 1260]
        elites = sorted_pop[:num_elite]
        non_elites = sorted_pop[num_elite:]

        # 3. Tạo thế hệ mới
        new_pop = []
        
        # 3.1. Copy Elites (Elitism)
        new_pop.extend(elites)
        
        # 3.2. Generate Mutants (Ngẫu nhiên hoàn toàn) [cite: 1263]
        mutants = np.random.rand(num_mutant, self.chromosome_len)
        new_pop.extend(mutants)
        
        # 3.3. Crossover (Biased) [cite: 1267]
        # Lai ghép giữa 1 Elite và 1 Non-Elite
        for _ in range(num_crossover):
            elite_parent = elites[np.random.randint(0, num_elite)]
            non_elite_parent = non_elites[np.random.randint(0, len(non_elites))]
            
            # Parameterized Uniform Crossover
            offspring = np.where(np.random.rand(self.chromosome_len) < rho,
                                    elite_parent,
                                    non_elite_parent)
            new_pop.append(offspring)
        
        self.population = np.array(new_pop)

    def solve(self):
        # Backward compatibility if needed, but main.py uses evolve()
        self.initialize()
        self.evolve()
        return self.best_solution, self.best_fitness