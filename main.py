import yaml
import numpy as np
from src.utils.data_loader import load_dataset
from src.decoder.decoder import Decoder
from src.algorithms.brkga import BRKGAAlgorithm
from src.algorithms.de import DEAlgorithm
from src.algorithms.pso import PSOAlgorithm
# from src.algorithms.ga import GAAlgorithm

def main():
    print("=== 3D Bin Packing Problem - Unified Evolutionary Framework ===")
    
    # 1. Load Config
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please run the generator script properly.")
        return

    # 2. Load Data (Giả lập nếu không có file thực)
    # Trong thực tế, load từ data/class1.txt
    items = load_dataset(config['problem']['dataset_path'])
    bin_dims = tuple(config['problem']['bin_dimensions'])
    
    print(f"Loaded {len(items)} items.")
    print(f"Bin Dimensions: {bin_dims}")

    # 3. Initialize Decoder (The Bridge)
    # Decoder chứa logic 'Placement Procedure' và 'Maximal Space' [cite: 1210]
    decoder = Decoder(items, bin_dims)
    
    # 4. Select Strategy
    algo_name = config['algorithm']['name']
    print(f"Running Algorithm: {algo_name.upper()}")
    
    solver = None
    if algo_name == 'brkga':
        solver = BRKGAAlgorithm(decoder, config)
    elif algo_name == 'de':
        solver = DEAlgorithm(decoder, config)
    elif algo_name == 'pso':
        solver = PSOAlgorithm(decoder, config)
    else:
        print(f"Algorithm {algo_name} not implemented yet.")
        return
    
    # 5. Run Optimization
    best_solution, best_fitness = solver.solve()
    
    # 6. Decode Final Result
    final_bins = decoder.decode(best_solution)
    print(f"\nOptimization Complete.")
    print(f"Total Bins Used: {len(final_bins)}")
    print(f"Best Fitness (aNB): {best_fitness:.4f}")

if __name__ == "__main__":
    main()