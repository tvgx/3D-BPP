import random
import pickle

from src.domain import Item

def generate_dataset(num_items=50, save_path="dataset.pkl"):
    # Bin Constraints
    BIN_DIMS = (100, 100, 100)
    MAX_WEIGHT = 500 # Example max weight
    
    items = []
    # Generate random items typical for logistics
    for i in range(num_items):
        # Dimensions between 10% and 40% of bin size
        d = random.randint(10, 40)
        w = random.randint(10, 40)
        h = random.randint(10, 40)
        # Weight proportional to volume + noise
        vol = d * w * h
        weight = max(1, int(vol * 0.001 * random.uniform(0.8, 1.2)))
        items.append(Item(i, d, w, h, weight))
        
    data = {
        "bin_dims": BIN_DIMS,
        "max_weight": MAX_WEIGHT,
        "items": items
    }
    
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
        
    print(f"Dataset generated at {save_path} with {num_items} items.")
    return data

if __name__ == "__main__":
    generate_dataset()