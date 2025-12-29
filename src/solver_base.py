from src.decoder import HeuristicDecoder

class Solver:
    def __init__(self, items, bin_dims, max_weight, pop_size, generations):
        self.items = items
        self.bin_dims = bin_dims
        self.max_weight = max_weight
        self.pop_size = pop_size
        self.generations = generations
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
        
        total_vol = sum([i[0].dims[0]*i[0].dims[1]*i[0].dims[2] for b in bins for i in b.items])
        capacity = n_bins * (self.bin_dims[0] * self.bin_dims[1] * self.bin_dims[2])
        fill_rate = total_vol / capacity
        
        return n_bins + (1.0 - fill_rate)
