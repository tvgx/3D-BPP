import yaml
import os
import argparse
import numpy as np
from src.utils.mendeley_parser import MendeleyParser
from src.utils.pickle_parser import PickleParser
from src.utils.config_loader import ConfigLoader
from src.core.representation import RandomKeyRepresentation
from src.evaluation.packing_simulator import PackingSimulator
from src.evaluation.fitness import FitnessEvaluator
from src.utils.visualization import Visualizer

# Algorithms
from src.algorithms.de import DE_SHADE
from src.algorithms.es import CMA_ES
from src.algorithms.mfea import MFEA
from src.algorithms.brkga import BRKGA
from src.algorithms.ga import GA

class ProblemDecoder:
    """
    Adapter class to bridge Chromosome -> Fitness for Algorithms.
    """
    def __init__(self, items, bin_dims):
        self.items = items
        self.bin_dims = bin_dims # (D, W, H)

    def decode(self, chromosome):
        """
        Decodes a chromosome into a list of packed Bins.
        """
        # 1. Decode Random Keys -> Packing Plan
        sorted_indices, rotations, heuristics = RandomKeyRepresentation.decode(chromosome, len(self.items))
        
        # 2. Run Placement Heuristic (Simulator)
        packed_bins = PackingSimulator.pack(
            self.items, 
            self.bin_dims, 
            sorted_indices, 
            rotations, 
            heuristics
        )
        return packed_bins

    def get_fitness(self, chromosome):
        bins = self.decode(chromosome)
        # Using simple capacity volume as default if not available, mainly for ANB calculation logic which might use it
        return FitnessEvaluator.calculate_anb(bins) # Updated to use new calculate_anb signature
    
    # For MFEA (Task interface)
    def evaluate(self, individual):
        # MFEA passes an Individual object, others might pass chromosome directly
        chrom = individual.chromosome if hasattr(individual, 'chromosome') else individual
        return self.get_fitness(chrom)
    
    @property
    def num_items(self):
        return len(self.items)

def main():
    print("=== 3D Bin Packing Problem - Optimization System ===")
    
    # 0. Setup Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="de", choices=["de", "es", "mfea", "brkga", "ga", "nsga2"], help="Algorithm to run")
    parser.add_argument("--data", type=str, default="data/instances/mendeley_v2/sample.txt", help="Path to data file")
    parser.add_argument("--config", type=str, default=None, help="Path to custom config file (optional)")
    parser.add_argument("--viz", action="store_true", help="Visualize result")
    args = parser.parse_args()

    # 1. Load Config
    # Strategy:
    # a. Load Base Config (or Algo specific config as base)
    # b. Load Custom Config if provided (Override)
    
    print(f"Loading Configuration for: {args.algo.upper()}")
    algo_config = ConfigLoader.get_algorithm_config(args.algo)
    
    if args.config and os.path.exists(args.config):
        print(f"Loading Custom Config: {args.config}")
        custom_config = ConfigLoader.load_config(args.config)
        config = ConfigLoader.merge_configs(algo_config, custom_config)
    else:
        config = algo_config
        
    # Ensure critical sections exist if config file was empty or missing
    if 'algorithm' not in config: config['algorithm'] = {'name': args.algo, 'generations': 50, 'pop_size': 30}
    if 'problem' not in config: config['problem'] = {'bin_dimensions': [100, 100, 100]}

    # 2. Load Data
    print(f"Loading data from: {args.data}")
    
    if args.data.endswith(".pkl"):
        items, bin_dims_from_file = PickleParser.parse(args.data)
    else:
        items, bin_dims_from_file = MendeleyParser.parse(args.data)
    
    if not items:
        print("Error: No items loaded. Check data file path and format.")
        return

    # Use bin dims from file if available, else config
    if bin_dims_from_file != (0,0,0):
        print(f"Bin Dimensions from file: {bin_dims_from_file}")
        bin_dims = bin_dims_from_file
    else:
        bin_dims = tuple(config['problem']['bin_dimensions'])
        print(f"Using default Bin Dimensions: {bin_dims}")

    print(f"Total Items: {len(items)}")

    # 3. Setup Decoder (The 'Task')
    decoder = ProblemDecoder(items, bin_dims)

    # 4. Initialize Algorithm
    algo_name = args.algo.lower()
    print(f"Initializing Algorithm: {algo_name.upper()}")
    
    solver = None
    if algo_name == 'de':
        solver = DE_SHADE(config['algorithm'], decoder)
    elif algo_name == 'es':
        solver = CMA_ES(config['algorithm'], decoder)
    elif algo_name == 'mfea':
        # MFEA typically needs multiple tasks.
        # For demo, we can create 2 variations (e.g. subset of items or same task duplicated)
        # Or just run it as single task (MFEA reduces to EA)
        print("Note: Running MFEA in single-task mode for demo.")
        solver = MFEA(config['algorithm'], [decoder]) 
    elif algo_name == 'brkga':
        solver = BRKGA(config['algorithm'], decoder)
    elif algo_name == 'ga':
        solver = GA(config['algorithm'], decoder) 
        
    else:
        print(f"Unknown algorithm: {algo_name}")
        return

    # 5. Run Optimization
    print("Starting optimization...")
    solver.initialize()
    solver.evolve()
    
    # 6. Retrieve Best Result
    # Adapting return types
    if algo_name == 'mfea':
        # For MFEA, get best from population
        best_ind = solver.population.get_best()
        best_chrom = best_ind.chromosome
        best_fitness = best_ind.fitness if best_ind.fitness != float('inf') else decoder.get_fitness(best_chrom)
    else:
        best_chrom = solver.best_solution
        best_fitness = solver.best_fitness
        
    print(f"\nOptimization Finished.")
    print(f"Best Fitness (aNB): {best_fitness:.4f}")
    
    # 7. Decode and Visualize
    final_bins = decoder.decode(best_chrom)
    print(f"Bins Used: {len(final_bins)}")
    
    for i, b in enumerate(final_bins):
        print(f"  Bin {i}: Fill Rate = {b.get_fill_rate():.2%}, Items = {len(b.items)}")

    if args.viz:
        print("Generating visualization...")
        output_file = "result_visualization.html"
        Visualizer.visualize_solution(final_bins, save_path=output_file)
        print(f"Visualization saved to: {os.path.abspath(output_file)}")

if __name__ == "__main__":
    main()
