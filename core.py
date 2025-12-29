import numpy as np


class Item:
    def __init__(self, id, d, w, h, weight):
        self.id = id
        self.dims = (d, w, h) # L, W, H
        self.weight = weight
    
    def __repr__(self):
        return f"Item({self.id}, dims={self.dims}, weight={self.weight})"

class Bin:
    def __init__(self, dims, max_weight):
        self.dims = dims # (D, W, H)
        self.max_weight = max_weight
        self.current_weight = 0
        self.items = [] # List of (item, position, rotation_type)
    
    def can_fit(self, item, pos, dims_rot):
        # 1. Weight Check
        if self.current_weight + item.weight > self.max_weight:
            return False
            
        # 2. Boundary Check
        x, y, z = pos
        d, w, h = dims_rot
        bd, bw, bh = self.dims
        if x + d > bd or y + w > bw or z + h > bh:
            return False
            
        # 3. Intersection Check
        for exist_item, exist_pos, exist_dim in self.items:
            ex, ey, ez = exist_pos
            ed, ew, eh = exist_dim
            
            # Check overlap
            if (x < ex + ed and x + d > ex and 
                y < ey + ew and y + w > ey and 
                z < ez + eh and z + h > ez):
                return False
        return True

    def add_item(self, item, pos, dims_rot):
        self.items.append((item, pos, dims_rot))
        self.current_weight += item.weight

class HeuristicDecoder:
    @staticmethod
    def decode(chromosome, items, bin_dims, max_weight):
        # chromosome length = 2 * n (Priority + Rotation)
        n = len(items)
        if len(chromosome) < 2*n:
            # Handle mismatch if resizing happens
            chromosome = np.resize(chromosome, 2*n)
            
        priorities = chromosome[:n]
        rotations = chromosome[n:]
        
        # Sort by priority
        sorted_indices = np.argsort(priorities)[::-1] # Descending
        
        bins = []
        
        # Helper to find best position in a bin
        def find_best_position(bin_obj, item, item_dims):
            # Candidate points: (0,0,0) and corners extended from existing items
            candidates = [(0,0,0)]
            for _, pos, dim in bin_obj.items:
                candidates.append((pos[0] + dim[0], pos[1], pos[2]))
                candidates.append((pos[0], pos[1] + dim[1], pos[2]))
                candidates.append((pos[0], pos[1], pos[2] + dim[2]))
            
            # Sort candidates (Bottom-Back-Left) -> Z then Y then X
            candidates = sorted(list(set(candidates)), key=lambda p: (p[2], p[1], p[0]))
            
            for pos in candidates:
                if bin_obj.can_fit(item, pos, item_dims):
                    return pos
            return None

        # Packing Loop
        for idx in sorted_indices:
            item = items[idx]
            rot_idx = int(rotations[idx] * 6) % 6
            item_dims = get_rotated_dims(item.dims, rot_idx)
            
            placed = False
            for b in bins:
                pos = find_best_position(b, item, item_dims)
                if pos:
                    b.add_item(item, pos, item_dims)
                    placed = True
                    break
            
            if not placed:
                new_bin = Bin(bin_dims, max_weight)
                # Try (0,0,0) in new bin
                if new_bin.can_fit(item, (0,0,0), item_dims):
                    new_bin.add_item(item, (0,0,0), item_dims)
                    bins.append(new_bin)
                else:
                    # Item too big/heavy for empty bin (Should not happen in generated data but safe to handle)
                    pass 
                    
        return bins

def get_rotated_dims(dims, rot_type):
    d, w, h = dims
    perms = [
        (d, w, h), (d, h, w),
        (w, d, h), (w, h, d),
        (h, d, w), (h, w, d)
    ]
    return perms[rot_type]