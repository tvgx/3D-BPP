from src.decoder import HeuristicDecoder

class Solver:
    def __init__(self, items, bin_dims, max_weight, pop_size, generations, fitness_k=2.0):
        self.items = items
        self.bin_dims = bin_dims
        self.max_weight = max_weight
        self.pop_size = pop_size
        self.generations = generations
        self.fitness_k = fitness_k
        self.history = []
        self.dim = 2 * len(items) 
        self.best_solution = None
        
    @property
    def best_bins(self):
        if self.best_solution is None: return []
        return HeuristicDecoder.decode(self.best_solution, self.items, self.bin_dims, self.max_weight)

    def evaluate(self, chromosome):
        bins = HeuristicDecoder.decode(chromosome, self.items, self.bin_dims, self.max_weight)
        n_bins = len(bins)
        if n_bins == 0: return float('inf')
        
        # Falkenauer's Fitness: favor fuller bins
        # F = n_bins + (1 - mean_fill^k)
        
        total_vol = 0
        fill_rates = []
        bin_vol = self.bin_dims[0] * self.bin_dims[1] * self.bin_dims[2]
        
        for b in bins:
             b_vol = sum([i[0].dims[0]*i[0].dims[1]*i[0].dims[2] for i in b.items])
             fill_rates.append((b_vol / bin_vol) ** self.fitness_k)
             
        mean_fill_k = sum(fill_rates) / n_bins
        
        return n_bins + (1.0 - mean_fill_k)
