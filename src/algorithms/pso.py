import numpy as np
from src.algorithms.base_solver import BaseSolver

class PSOAlgorithm(BaseSolver):
    """
    Particle Swarm Optimization (PSO).
    Dựa trên tài liệu[cite: 9].
    Particles bay trong không gian hypercube [0, 1]^3N.
    """
    def solve(self):
        params = self.config['algorithm']['pso']
        w = params['w']   # Quán tính [cite: 47]
        c1 = params['c1'] # Nhận thức
        c2 = params['c2'] # Xã hội

        # Khởi tạo vận tốc
        velocities = np.random.rand(self.pop_size, self.chromosome_len) * 0.1
        
        # Pbest (Cá thể tốt nhất của từng hạt)
        pbest_pos = np.copy(self.population)
        pbest_val = np.array([float('inf')] * self.pop_size)
        
        # Gbest (Tốt nhất toàn đàn)
        gbest_pos = None
        gbest_val = float('inf')

        for gen in range(self.config['algorithm']['generations']):
            # Đánh giá fitness
            for i in range(self.pop_size):
                fit = self.decoder.get_fitness(self.population[i])
                
                # Cập nhật Pbest [cite: 26]
                if fit < pbest_val[i]:
                    pbest_val[i] = fit
                    pbest_pos[i] = self.population[i].copy()
                
                # Cập nhật Gbest [cite: 29]
                if fit < gbest_val:
                    gbest_val = fit
                    gbest_pos = self.population[i].copy()
            
            print(f"Gen {gen}: Best Fitness aNB = {gbest_val:.4f}")
            
            # Cập nhật Vận tốc và Vị trí [cite: 46]
            r1 = np.random.rand(self.pop_size, self.chromosome_len)
            r2 = np.random.rand(self.pop_size, self.chromosome_len)
            
            velocities = (w * velocities + 
                          c1 * r1 * (pbest_pos - self.population) + 
                          c2 * r2 * (gbest_pos - self.population))
            
            self.population = self.population + velocities
            
            # Quan trọng: Kẹp giá trị về [0, 1] để giữ đúng tính chất Random Key
            self.population = np.clip(self.population, 0, 1)

        return gbest_pos, gbest_val