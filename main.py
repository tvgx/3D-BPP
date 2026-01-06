import argparse
import pickle
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from src.domain import Item
from elhedhli_parser import ElhedhliParser
from src.solver_ga import GA
from src.solver_es import ES
from src.solver_aco import ACO
from src.solver_pso import PSO
from src.domain import Item
from src.visualization import plot_convergence, visualize_3d_packing, visualize_single_bin, create_color_map

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ga", "es", "aco", "pso"], default="ga", help="Algorithm to run")
    parser.add_argument("--gen", type=int, default=config.get("ga", {}).get("generations", 100), help="Generations")
    parser.add_argument("--data", type=str, default=None, help="Path to data file (.txt or .pkl)")
    parser.add_argument("--viz", action="store_true", help="Generate visualizations (convergence plot and 3D packing)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of items to load (for large datasets)")
    args = parser.parse_args()
    
    # Process Data Logic ...
    # Process Data Logic
    if args.data and os.path.isdir(args.data):
        files_to_process = []
        for root, dirs, files in os.walk(args.data):
            for f in files:
                if f.endswith(".txt") or f.endswith(".pkl"):
                    files_to_process.append(os.path.join(root, f))
        
        files_to_process.sort()
        print(f"Found {len(files_to_process)} test cases in {args.data}")
        
        for path in files_to_process:
            print(f"\n{'='*60}")
            print(f"PROCESSING: {path}")
            print(f"{'='*60}")
            try:
                run_single_file(path, args.algo, args.gen, config, args.viz, args.limit)
            except Exception as e:
                print(f"Error processing {path}: {e}")
    else:
        run_single_file(args.data, args.algo, args.gen, config, args.viz, args.limit)

def run_single_file(data_path, algo, gen, config, enable_viz=False, limit=None):
    items, bin_dims, max_weight = load_data(data_path, config, limit=limit)
    if not items: return

    # Create results directory
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Get instance name from data path for organized storage
    instance_name = "default"
    if data_path:
        instance_name = os.path.splitext(os.path.basename(data_path))[0]
        # Remove path separators if any
        instance_name = instance_name.replace("/", "_").replace("\\", "_")
    
    # Create subdirectory for this algorithm and instance
    algo_instance_dir = os.path.join(results_dir, f"{algo}_{instance_name}")
    os.makedirs(algo_instance_dir, exist_ok=True)

    fitness_k = config.get("problem", {}).get("fitness_k", 2.0)
    
    solver = None
    if algo == "ga":
        pop_size = config.get("ga", {}).get("pop_size", 100)
        solver = GA(items, bin_dims, max_weight, pop_size=pop_size, generations=gen, fitness_k=fitness_k)
    elif algo == "es":
        # ES uses internal Mu/Lambda from config usually, but passing gen
        solver = ES(items, bin_dims, max_weight, pop_size=0, generations=gen, fitness_k=fitness_k)
    elif algo == "aco":
        pop_size = config.get("aco", {}).get("pop_size", 100)
        solver = ACO(items, bin_dims, max_weight, pop_size=pop_size, generations=gen, fitness_k=fitness_k)
    elif algo == "pso":
        pop_size = config.get("pso", {}).get("pop_size", 100)
        solver = PSO(items, bin_dims, max_weight, pop_size=pop_size, generations=gen, fitness_k=fitness_k)
    
    if solver is None:
        raise ValueError(f"Unsupported algorithm '{algo}'. Expected one of: ga, es, aco, pso.")

    if not hasattr(solver, 'run') or solver.run is None:
        raise AttributeError(f"Solver for algorithm '{algo}' does not implement a 'run' method.")

    history = solver.run()
    
    # Get best solution
    bins = solver.best_bins 
    print_solution(bins)
    
    # Save solution summary to text file
    solution_file = os.path.join(algo_instance_dir, "solution_summary.txt")
    with open(solution_file, "w") as f:
        f.write(f"Algorithm: {algo.upper()}\n")
        f.write(f"Instance: {instance_name}\n")
        f.write(f"Generations: {gen}\n")
        f.write(f"Number of bins used: {len(bins)}\n")
        f.write(f"Number of items packed: {sum([len(b.items) for b in bins])}\n")
        f.write(f"Final fitness: {history[-1] if history else 'N/A'}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("Detailed Packing Solution:\n")
        f.write("="*60 + "\n\n")
        
        # Write the same format as print_solution
        f.write(f"{'id':>4}  {'bin-loc':>9}  {'orientation':>13}  {'x':>3}  {'y':>3}  {'z':>3}  {'x\'':>4}  {'y\'':>4}  {'z\'':>4}  {'weight':>8}\n")
        f.write(f"{'-'*4}  {'-'*9}  {'-'*13}  {'-'*3}  {'-'*3}  {'-'*3}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*8}\n")
        
        for b_idx, b in enumerate(bins):
            for (item, pos, dims) in b.items:
                orient = match_orientation(item.dims, dims)
                f.write(f"{item.id:>4}  {b_idx+1:>9}  {orient:>13}  {pos[0]:>3}  {pos[1]:>3}  {pos[2]:>3}  {dims[0]:>4}  {dims[1]:>4}  {dims[2]:>4}  {item.weight:>8.0f}\n")
    
    print(f"\nSolution summary saved to: {solution_file}")
    
    # Generate visualizations if requested
    if enable_viz:
        print("\n" + "="*60)
        print("Generating visualizations...")
        print("="*60)
        
        # Generate convergence plot
        plot_path = os.path.join(algo_instance_dir, f"convergence_plot_{algo}.png")
        plot_convergence(history, algo, save_path=plot_path)
        
        # Generate 3D packing visualization
        if bins:
            # Create consistent color map for all visualizations
            color_map = create_color_map(bins)
            
            # Visualize all bins in one figure
            viz_path = os.path.join(algo_instance_dir, f"result_visualization_{algo}.html")
            visualize_3d_packing(bins, bin_dims, algorithm_name=algo, save_path=viz_path)
            
            # Also create individual bin visualizations for first few bins
            # Use the same color map for consistency
            for bin_idx in range(min(3, len(bins))):
                single_bin_path = os.path.join(algo_instance_dir, f"result_visualization_bin_{bin_idx}_{algo}.html")
                visualize_single_bin(bins[bin_idx], bin_dims, bin_idx=bin_idx, 
                                     algorithm_name=algo, save_path=single_bin_path,
                                     color_map=color_map)
        
        print(f"\nAll results saved to: {algo_instance_dir}")
        print("Visualization complete!")

def load_data(path, config, limit=None):
    default_dims = tuple(config.get("problem", {}).get("default_bin_dims", [1200, 1200, 1200]))
    default_max_weight = config.get("problem", {}).get("default_max_weight", 10000)

    if path and os.path.exists(path):
        if path.endswith(".txt"):
            items, bin_dims, max_weight = ElhedhliParser.parse(path)
            if bin_dims == (0,0,0): bin_dims = default_dims
            if max_weight == float('inf'): max_weight = default_max_weight
            if limit and len(items) > limit:
                items = items[:limit]
                print(f"Limited to first {limit} items from dataset")
            return items, bin_dims, max_weight
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "items" in data:
                    items = data["items"]
                    if limit and len(items) > limit:
                        items = items[:limit]
                        print(f"Limited to first {limit} items from dataset")
                    return items, data["bin_dims"], data["max_weight"]
                elif isinstance(data, pd.DataFrame):
                    items = []
                    max_rows = limit if limit else len(data)
                    if limit:
                        print(f"Loading first {limit} items from DataFrame (total: {len(data)})")
                    for i, row in data.head(max_rows).iterrows():
                        w = row['width'] if 'width' in data.columns else row.get('w', 10)
                        d = row['depth'] if 'depth' in data.columns else row.get('d', 10)
                        h = row['height'] if 'height' in data.columns else row.get('h', 10)
                        wt = row['weight'] if 'weight' in data.columns else row.get('weight', 1)
                        items.append(Item(i, int(d), int(w), int(h), float(wt)))
                    return items, default_dims, default_max_weight
                else:
                     return [], (0,0,0), 0
            except:
                 return [], (0,0,0), 0

def print_solution(bins):
    total_bins = len(bins)
    total_items = sum([len(b.items) for b in bins])
    print(f"# Number of bins used: {total_bins}")
    print(f"# Number of cases packed: {total_items}")
    
    print(" ")
    print(f"{'id':>4}  {'bin-loc':>9}  {'orientation':>13}  {'x':>3}  {'y':>3}  {'z':>3}  {'x\'':>4}  {'y\'':>4}  {'z\'':>4}  {'weight':>8}")
    print(f"{'-'*4}  {'-'*9}  {'-'*13}  {'-'*3}  {'-'*3}  {'-'*3}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*8}")
    
    for b_idx, b in enumerate(bins):
        for (item, pos, dims) in b.items:
            orient = match_orientation(item.dims, dims)
            print(f"{item.id:>4}  {b_idx+1:>9}  {orient:>13}  {pos[0]:>3}  {pos[1]:>3}  {pos[2]:>3}  {dims[0]:>4}  {dims[1]:>4}  {dims[2]:>4}  {item.weight:>8.0f}")

def match_orientation(orig, curr):
    d, w, h = orig
    perms = [
        (d, w, h), (d, h, w),
        (w, d, h), (w, h, d),
        (h, d, w), (h, w, d)
    ]
    for i, p in enumerate(perms):
        if p == curr:
            return i + 1 
    return 0

if __name__ == "__main__":
    main()