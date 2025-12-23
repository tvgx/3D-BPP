import numpy as np

class RandomKeyRepresentation:
    """
    Handles the encoding and decoding of solutions using Random Keys.
    Structure: [Order (N) | Rotation (N) | Heuristic (N)]
    """
    
    @staticmethod
    def decode(chromosome, items_count):
        """
        Decodes the chromosome into:
        1. Packing Order (Permutation)
        2. Rotation (0-5)
        3. Placement Heuristic (e.g., 0 or 1)
        
        Args:
            chromosome: 1D numpy array of size 3 * N.
            items_count: Number of items (N).
            
        Returns:
            Tuple(sorted_item_indices, rotations, heuristics)
        """
        n = items_count
        if len(chromosome) != 3 * n:
             # In some cases (like variable length), this might differ, but for standard 3D-BPP RK, it's fixed.
             raise ValueError(f"Chromosome length {len(chromosome)} does not match 3 * num_items ({3*n}).")

        # Split chromosome into 3 parts
        # Part 1: Packing Order (r1). Range [0, n-1] index mapping.
        # We sort the indices based on r1 values.
        r1 = chromosome[:n]
        sorted_item_indices = np.argsort(r1) 

        # Part 2: Rotation (r2). Range [0, 1] -> [0, 5]
        # Maps [0, 1] to integers [0, 5].
        r2 = chromosome[n:2*n]
        rotations = np.floor(r2 * 6).astype(int)
        rotations = np.clip(rotations, 0, 5)

        # Part 3: Placement Heuristic (r3). Range [0, 1] -> [0, 1]
        # Maps [0, 1] to heuristic IDs (e.g. 0=EP, 1=DBL or similar)
        r3 = chromosome[2*n:]
        heuristics = np.where(r3 < 0.5, 0, 1).astype(int)
        
        return sorted_item_indices, rotations, heuristics
