import numpy as np
from src.domain import Bin
from src.constraints import ConstraintChecker

class HeuristicDecoder:
    @staticmethod
    def decode(chromosome, items, bin_dims, max_weight):
        # chromosome length = 2 * n (Priority + Rotation)
        n = len(items)
        if len(chromosome) < 2*n:
            chromosome = np.resize(chromosome, 2*n)
            
        priorities = chromosome[:n]
        rotations = chromosome[n:]
        
        # Sort by priority
        sorted_indices = np.argsort(priorities)[::-1] 
        
        bins = []
        
        # Helper to find best position in a bin
        def find_best_position(bin_obj, item, item_dims):
            # Candidate points: (0,0,0) and corners extended from existing items
            candidates = [(0,0,0)]
            for _, pos, dim in bin_obj.items:
                candidates.append((pos[0] + dim[0], pos[1], pos[2]))
                candidates.append((pos[0], pos[1] + dim[1], pos[2]))
                candidates.append((pos[0], pos[1], pos[2] + dim[2]))
            
            # Sort candidates (Bottom-Back-Left)
            candidates = sorted(list(set(candidates)), key=lambda p: (p[2], p[1], p[0]))
            
            for pos in candidates:
                if ConstraintChecker.can_fit(bin_obj, item, pos, item_dims):
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
                if ConstraintChecker.can_fit(new_bin, item, (0,0,0), item_dims):
                    new_bin.add_item(item, (0,0,0), item_dims)
                    bins.append(new_bin)
                # Else item fits nowhere (skipped or error case)
                    
        return bins

def get_rotated_dims(dims, rot_type):
    d, w, h = dims
    perms = [
        (d, w, h), (d, h, w),
        (w, d, h), (w, h, d),
        (h, d, w), (h, w, d)
    ]
    return perms[rot_type]
