import argparse
import pickle
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

from data_gen import generate_dataset
from elhedhli_parser import ElhedhliParser
from src.solver_ga import GA
from src.solver_es import ES
from src.domain import Item

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, choices=["ga", "es"], default="ga", help="Algorithm to run")
    parser.add_argument("--gen", type=int, default=config.get("ga", {}).get("generations", 100), help="Generations")
    parser.add_argument("--data", type=str, default=None, help="Path to data file (.txt or .pkl)")
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
                run_single_file(path, args.algo, args.gen, config)
            except Exception as e:
                print(f"Error processing {path}: {e}")
    else:
        run_single_file(args.data, args.algo, args.gen, config)

def run_single_file(data_path, algo, gen, config):
    items, bin_dims, max_weight = load_data(data_path, config)
    if not items: return

    solver = None
    if algo == "ga":
        pop_size = config.get("ga", {}).get("pop_size", 100)
        solver = GA(items, bin_dims, max_weight, pop_size=pop_size, generations=gen)
    elif algo == "es":
        # ES uses internal Mu/Lambda from config usually, but passing gen
        solver = ES(items, bin_dims, max_weight, pop_size=0, generations=gen)
        
    history = solver.run()
    
    # Get best solution
    bins = solver.best_bins 
    print_solution(bins)

def load_data(path, config):
    default_dims = tuple(config.get("problem", {}).get("default_bin_dims", [1200, 1200, 1200]))
    default_max_weight = config.get("problem", {}).get("default_max_weight", 10000)

    if path and os.path.exists(path):
        if path.endswith(".txt"):
            items, bin_dims, max_weight = ElhedhliParser.parse(path)
            if bin_dims == (0,0,0): bin_dims = default_dims
            if max_weight == float('inf'): max_weight = default_max_weight
            return items, bin_dims, max_weight
        elif path.endswith(".pkl"):
            try:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                if isinstance(data, dict) and "items" in data:
                    return data["items"], data["bin_dims"], data["max_weight"]
                elif isinstance(data, pd.DataFrame):
                    items = []
                    for i, row in data.iterrows():
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
    
    # Fallback/Generate
    # If using generated data, we rely on data_gen default
    try:
        with open("dataset.pkl", "rb") as f:
             data = pickle.load(f)
        return data["items"], data["bin_dims"], data["max_weight"]
    except:
        data = generate_dataset()
        return data["items"], data["bin_dims"], data["max_weight"]

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