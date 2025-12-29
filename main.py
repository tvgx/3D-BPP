from data_gen import generate_dataset
from solvers import GA, ES
from core import Item
from elhedhli_parser import ElhedhliParser
import os
import pickle
import pandas as pd
import argparse
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ga", "es"], default="ga", help="Algorithm to run")
    parser.add_argument("--gen", type=int, default=100, help="Generations")
    parser.add_argument("--data", type=str, default=None, help="Path to data file (.txt or .pkl)")
    args = parser.parse_args()
    
    # Batch Processing or Single File
    if args.data and os.path.isdir(args.data):
        # Process all txt files in directory
        files = [f for f in os.listdir(args.data) if f.endswith(".txt") or f.endswith(".pkl")]
        files.sort()
        for f in files:
            path = os.path.join(args.data, f)
            print(f"\nPROCESSING: {path}")
            run_single_file(path, args.algo, args.gen)
    else:
        # Single file
        run_single_file(args.data, args.algo, args.gen)

def run_single_file(data_path, algo, gen):
    # Load Data
    items, bin_dims, max_weight = load_data(data_path)
    if not items: return

    # print(f"Running {algo.upper()} on {len(items)} items...")
    
    solver = None
    if algo == "ga":
        solver = GA(items, bin_dims, max_weight, pop_size=100, generations=gen)
    elif algo == "es":
        solver = ES(items, bin_dims, max_weight, pop_size=20, generations=gen)
        
    history = solver.run()
    
    # Get best solution
    best_chrom = solver.best_solution
    bins = solver.best_bins # Need to expose this from Solver/GA
    
    # Print formatted output
    print_solution(bins)

def load_data(path):
    # Refactored load logic
    if path and os.path.exists(path):
        if path.endswith(".txt"):
            items, bin_dims, max_weight = ElhedhliParser.parse(path)
            if bin_dims == (0,0,0): bin_dims = (1200, 1200, 1200)
            if max_weight == float('inf'): max_weight = 10000 
            return items, bin_dims, max_weight
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "items" in data:
                    return data["items"], data["bin_dims"], data["max_weight"]
                elif isinstance(data, pd.DataFrame):
                    bin_dims = (1200, 1200, 1200)
                    max_weight = 10000 
                    items = []
                    for i, row in data.iterrows():
                        w = row['width'] if 'width' in data.columns else row.get('w', 10)
                        d = row['depth'] if 'depth' in data.columns else row.get('d', 10)
                        h = row['height'] if 'height' in data.columns else row.get('h', 10)
                        wt = row['weight'] if 'weight' in data.columns else row.get('weight', 1)
                        items.append(Item(i, int(d), int(w), int(h), float(wt)))
                    return items, bin_dims, max_weight
                else:
                    return [], (0,0,0), 0
            except:
                return [], (0,0,0), 0
    
    # Default fallback logic omitted for batch mode simplicity, or keep if needed
    # ...
    return [], (0,0,0), 0

def print_solution(bins):
    total_bins = len(bins)
    total_items = sum([len(b.items) for b in bins])
    print(f"# Number of bins used: {total_bins}")
    print(f"# Number of cases packed: {total_items}")
    # print(f"# Objective value: ...") 
    
    print(" ")
    print(f"{'id':>4}  {'bin-loc':>9}  {'orientation':>13}  {'x':>3}  {'y':>3}  {'z':>3}  {'x\'':>4}  {'y\'':>4}  {'z\'':>4}  {'weight':>8}")
    print(f"{'-'*4}  {'-'*9}  {'-'*13}  {'-'*3}  {'-'*3}  {'-'*3}  {'-'*4}  {'-'*4}  {'-'*4}  {'-'*8}")
    
    for b_idx, b in enumerate(bins):
        for (item, pos, dims) in b.items:
            # Orientation? We have item.dims (original) and dims (packed).
            # Need to infer orientation index (1-6). 
            # Simple heuristic matching
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
            return i + 1 # 1-based index
    return 0


if __name__ == "__main__":
    main()