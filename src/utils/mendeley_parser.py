import os
from src.evaluation.packing_simulator import Item

class MendeleyParser:
    """
    Parser for Mendeley Data (v2) 3D-BPP instances.
    """
    
    @staticmethod
    def parse(data_path):
        """
        Parses a file in the format:
        Problem: ...
        Bin: D W H
        # Comments
        ID, L, W, H, Weight, Quantity, RotationConstraint
        """
        items = []
        bin_dims = (0, 0, 0)
        
        try:
            with open(data_path, 'r') as f:
                lines = [l.strip() for l in f.readlines()]
            
            # Filter empty lines
            lines = [l for l in lines if l]
            
            # Parse Header
            for i, line in enumerate(lines):
                if line.startswith("Bin:"):
                    parts = line.replace("Bin:", "").replace(",", " ").split()
                    bin_dims = (int(parts[0]), int(parts[1]), int(parts[2]))
                    continue
                if line.startswith("#") or line.startswith("Problem:"):
                    continue
                
                if line[0].isdigit():
                    parts = line.replace(",", " ").split()
                    if len(parts) >= 6:
                        d, w, h = int(parts[1]), int(parts[2]), int(parts[3])
                        qty = int(parts[5])
                        
                        rot_constraint = 0 # Default All
                        if len(parts) > 6:
                            rot_constraint = int(parts[6])
                        
                        valid_rotations = list(range(6))
                        if rot_constraint == 1:
                            valid_rotations = [0, 2]
                        
                        for _ in range(qty):
                            items.append(Item(len(items), (d, w, h), valid_rotations))
        
        except Exception as e:
            print(f"Error parsing {data_path}: {e}")
            return [], (0,0,0)
            
        return items, bin_dims
