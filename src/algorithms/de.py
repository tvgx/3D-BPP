import numpy as np
from src.algorithms.base_solver import BaseSolver

class DEAlgorithm(BaseSolver):
    """
    Differential Evolution (DE).
    Dựa trên tài liệu[cite: 587].
    Phù hợp với Random Key Encoding vì hoạt động trên số thực.
    """
    def solve(self):
        params = self.config['algorithm']['de']
        F = params['differential_weight'] # [cite: 689]
        CR = params['crossover_rate']     # [cite: 688]

        for gen in range(self.config['algorithm']['generations']):
            self.evaluate_all()
            print(f"Gen {gen}: Best Fitness aNB = {self.best_fitness:.4f}")
            
            new_pop = np.copy(self.population)
            
            for i in range(self.pop_size):
                # 1. Mutation: Chọn r1, r2, r3 khác i [cite: 650]
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = self.population[np.random.choice(idxs, 3, replace=False)]
                
                # Vector đột biến V = r1 + F * (r2 - r3) [cite: 652]
                mutant_vector = r1 + F * (r2 - r3)
                # Đảm bảo giá trị nằm trong [0, 1] (Random Key constraint)
                mutant_vector = np.clip(mutant_vector, 0, 1)
                
                # 2. Crossover (Binomial) [cite: 662]
                cross_points = np.random.rand(self.chromosome_len) < CR
                # Đảm bảo ít nhất 1 gen được thay đổi
                k = np.random.randint(0, self.chromosome_len)
                cross_points[k] = True
                
                trial_vector = np.where(cross_points, mutant_vector, self.population[i])
                
                # 3. Selection [cite: 668]
                # DE đánh giá ngay lập tức (greedy selection)
                f_trial = self.decoder.get_fitness(trial_vector)
                if f_trial <= self.fitnesses[i]: # Minimize problem
                    new_pop[i] = trial_vector
            
            self.population = new_pop

        return self.best_solution, self.best_fitness