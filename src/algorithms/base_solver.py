import numpy as np

class BaseSolver:
    def __init__(self, decoder, config):
        self.decoder = decoder
        self.config = config
        self.items_count = len(decoder.items)
        
        # Độ dài nhiễm sắc thể = 3 * N [cite: 1285]
        self.chromosome_len = 3 * self.items_count
        
        # Khởi tạo quần thể ngẫu nhiên [0, 1]
        self.pop_size = config['algorithm']['pop_size_multiplier'] * self.items_count
        self.population = np.random.rand(self.pop_size, self.chromosome_len)
        self.fitnesses = np.array([float('inf')] * self.pop_size)
        
        self.best_solution = None
        self.best_fitness = float('inf')

    def evaluate_all(self):
        """Đánh giá fitness cho toàn bộ quần thể"""
        # Trong thực tế nên dùng Parallel processing [cite: 1405]
        for i in range(self.pop_size):
            fit = self.decoder.get_fitness(self.population[i])
            self.fitnesses[i] = fit
            
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_solution = self.population[i].copy()