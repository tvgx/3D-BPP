import numpy as np

class Item:
    def __init__(self, item_id, dims, valid_rotations=None):
        self.id = item_id
        # Standardize dims to (D, W, H)
        self.dims = tuple(dims) 
        self.volume = dims[0] * dims[1] * dims[2]
        # Default to all 6 rotations if not specified
        self.valid_rotations = valid_rotations if valid_rotations is not None else list(range(6))

    def get_dimension(self, rotation_type):
        """
        Returns dimensions based on rotation type (0-5).
        L, W, H = 0, 1, 2
        """
        d, w, h = self.dims
        # 6 orientations
        # 0: d, w, h (Original Z is Z)
        # 1: d, h, w (Original Y is Z)
        # 2: w, d, h (Original Z is Z) - Rotated 90 deg around Z
        # 3: w, h, d (Original X is Z)
        # 4: h, d, w (Original Y is Z)
        # 5: h, w, d (Original X is Z)
        
        rotations = [
            (d, w, h), (d, h, w),
            (w, d, h), (w, h, d),
            (h, d, w), (h, w, d)
        ]
        
        # If strict check is needed:
        # if rotation_type not in self.valid_rotations: raise ValueError(...)
        
        return rotations[rotation_type % 6]

class Space:
    """
    Represents a maximal empty space in the bin.
    """
    def __init__(self, min_coord, max_coord):
        self.min_coord = min_coord # (x, y, z)
        self.max_coord = max_coord # (X, Y, Z) (exclusive usually, or inclusive? Let's assume max_coord is P2)
        # Dimensions
        self.dims = tuple(M - m for M, m in zip(max_coord, min_coord))
        self.volume = self.dims[0] * self.dims[1] * self.dims[2]

class Bin:
    def __init__(self, bin_id, dims):
        self.id = bin_id
        self.dims = dims # (D, W, H)
        self.capacity = dims[0] * dims[1] * dims[2]
        self.items = [] # List of tuples (Item, position, rotation_dims)
        self.free_spaces = [Space((0,0,0), dims)]

    def get_fill_rate(self):
        used_vol = sum([item.volume for item, _, _ in self.items])
        return used_vol / self.capacity

class PackingSimulator:
    """
    Simulates the packing process using Maximal-Space Representation.
    """
    @staticmethod
    def difference_process(free_spaces, item_min, item_max):
        """
        Updates the list of free spaces by subtracting the placed item.
        Returns a new list of maximal empty spaces.
        """
        ix1, iy1, iz1 = item_min
        ix2, iy2, iz2 = item_max
        
        new_spaces = []
        
        for space in free_spaces:
            sx1, sy1, sz1 = space.min_coord
            sx2, sy2, sz2 = space.max_coord
            
            # Check for intersection
            # Two 3D boxes intersect if they overlap in all 3 dimensions
            if (ix2 <= sx1 or ix1 >= sx2 or 
                iy2 <= sy1 or iy1 >= sy2 or 
                iz2 <= sz1 or iz1 >= sz2):
                new_spaces.append(space)
                continue
            
            # If intersecting, generate new spaces
            # We generate up to 6 new spaces by cutting the current space with the planes of the item
            
            # 1. Right of item (x > ix2)
            if ix2 < sx2 and ix2 > sx1:
                new_spaces.append(Space((ix2, sy1, sz1), (sx2, sy2, sz2)))
            # 2. Left of item (x < ix1)
            if ix1 > sx1 and ix1 < sx2:
                new_spaces.append(Space((sx1, sy1, sz1), (ix1, sy2, sz2)))
            # 3. Back of item (y > iy2) - Note: standard y axis usually y+ is back/up in some contexts, let's assume y2 > y1
            if iy2 < sy2 and iy2 > sy1:
                new_spaces.append(Space((sx1, iy2, sz1), (sx2, sy2, sz2)))
            # 4. Front of item (y < iy1)
            if iy1 > sy1 and iy1 < sy2:
                new_spaces.append(Space((sx1, sy1, sz1), (sx2, iy1, sz2)))
            # 5. Top of item (z > iz2)
            if iz2 < sz2 and iz2 > sz1:
                new_spaces.append(Space((sx1, sy1, iz2), (sx2, sy2, sz2)))
            # 6. Bottom of item (z < iz1)
            if iz1 > sz1 and iz1 < sz2:
                new_spaces.append(Space((sx1, sy1, sz1), (sx2, sy2, iz1)))
                
        # Filter non-maximal spaces
        # A space i is contained in space j if i is a subset of j.
        # Max-Space definition requires removing all such subsets.
        
        # Sort by volume descending for efficiency (check larger against smaller is wrong way, 
        # usually check if smaller is inside larger)
        # So sort by volume descending: largest first. 
        # Then for each space, check if it is contained in any space BEFORE it? 
        # No, "contained in ANY other space".
        
        # Optimization: 
        # If we just generated them, many might be subsets.
        
        # Helper to check containment: A inside B?
        def is_contained(A, B):
            return (A.min_coord[0] >= B.min_coord[0] and A.max_coord[0] <= B.max_coord[0] and
                    A.min_coord[1] >= B.min_coord[1] and A.max_coord[1] <= B.max_coord[1] and
                    A.min_coord[2] >= B.min_coord[2] and A.max_coord[2] <= B.max_coord[2])

        # Filter
        final_spaces = []
        # Sort by volume descending implies we iterate from largest. 
        # The largest cannot be contained in any smaller space.
        # So if we keep the list sorted, we only need to check if current is contained in any ALREADY ACCEPTED larger space?
        # Yes, if we process from largest to smallest.
        
        new_spaces.sort(key=lambda s: s.volume, reverse=True)
        
        for i, space in enumerate(new_spaces):
            contained = False
            for kept_space in final_spaces:
                if is_contained(space, kept_space):
                    contained = True
                    break
            if not contained:
                final_spaces.append(space)
        
        return final_spaces

    @staticmethod
    def pack(items, bin_dims, packing_order, rotations, heuristics):
        """
        Packs items into bins.
        
        Args:
            items: List of Item objects.
            bin_dims: (D, W, H)
            packing_order: List of item indices to pack.
            rotations: List of rotation types (0-5) for each item (indexed by item ID or order?).
                       Usually indexed by original item index.
            heuristics: List of heuristics (0 or 1) for each item.
            
        Returns:
            List of Bin objects with placed items.
        """
        bins = [Bin(0, bin_dims)]
        
        for item_idx in packing_order:
            item = items[item_idx]
            rotation = rotations[item_idx]
            heuristic = heuristics[item_idx] # 0: BBL, 1: BLB
            
            # Get actual dimensions after rotation
            item_d, item_w, item_h = item.get_dimension(rotation)
            
            placed = False
            
            # Try to pack into existing bins
            for b in bins:
                best_space = None
                best_metric = float('inf') # We want to minimize metric (e.g. position coordinates)
                
                # Find valid spaces
                candidates = []
                for s in b.free_spaces:
                    if (s.dims[0] >= item_d and 
                        s.dims[1] >= item_w and 
                        s.dims[2] >= item_h):
                        candidates.append(s)
                
                if not candidates:
                    continue
                    
                # Apply Heuristic to choose best space
                # Heuristic 0 (BBL): Back-Bottom-Left -> Z, X, Y
                # Heuristic 1 (BLB): Back-Left-Bottom -> Z, Y, X
                # Note: Coordinate system (x, y, z).
                # Usually: x=Depth, y=Width, z=Height.
                
                # Sort candidates
                if heuristic == 0: # BBL
                    # Prioritize Z (Bottom), then X (Back), then Y (Left)
                    # wait, standard names:
                    # Bottom-Left-Back: Z, Y, X ?
                    # Let's use: Z primary.
                    candidates.sort(key=lambda s: (s.min_coord[2], s.min_coord[0], s.min_coord[1]))
                else: # BLB? or just different
                    # Prioritize Z, then Y, then X
                    candidates.sort(key=lambda s: (s.min_coord[2], s.min_coord[1], s.min_coord[0]))
                
                best_space = candidates[0]
                
                # Place item
                # Calculate placement coords (bottom-left-back of space)
                x, y, z = best_space.min_coord
                
                # Add to bin
                b.items.append((item, (x, y, z), (item_d, item_w, item_h)))
                
                # usage: difference_process(free_spaces, placed_item_min, placed_item_max)
                placed_min = (x, y, z)
                placed_max = (x + item_d, y + item_w, z + item_h)
                
                b.free_spaces = PackingSimulator.difference_process(b.free_spaces, placed_min, placed_max)
                placed = True
                break
            
            if not placed:
                # Open new bin
                new_bin = Bin(len(bins), bin_dims)
                bins.append(new_bin)
                
                # Place at 0,0,0
                x, y, z = 0, 0, 0
                new_bin.items.append((item, (x, y, z), (item_d, item_w, item_h)))
                
                placed_min = (x, y, z)
                placed_max = (x + item_d, y + item_w, z + item_h)
                
                new_bin.free_spaces = PackingSimulator.difference_process(new_bin.free_spaces, placed_min, placed_max)
                
        return bins
