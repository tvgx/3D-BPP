import re
from src.domain import Item

class ElhedhliParser:
    @staticmethod
    def parse(file_path):
        """
        Parses an Elhedhli format file.
        Returns: items, bin_dims, max_weight
        """
        bin_dims = (0, 0, 0)
        max_weight = float('inf')
        items = []
        
        with open(file_path, 'r') as f:
            lines = f.readlines()
            
        # 1. Parse Metadata
        for line in lines:
            if line.startswith("# Bin dimensions"):
                # # Bin dimensions (L * W * H): (1200,1200,1200)
                match = re.search(r'\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)', line)
                if match:
                    bin_dims = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            
            elif line.startswith("# Max weight"):
                # # Max weight: 
                # Check if there's a value
                parts = line.split(":")
                if len(parts) > 1 and parts[1].strip():
                    try:
                        max_weight = float(parts[1].strip())
                    except ValueError:
                        pass
        
        # 2. Parse Items
        # Find header line index
        start_idx = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("----"):
                start_idx = i + 1
                break
        
        if start_idx == 0:
            # Fallback if separator not found, search for header "id quantity"
             for i, line in enumerate(lines):
                if "id" in line and "quantity" in line:
                    start_idx = i + 1
                    # Check next line for separator?
                    if i+1 < len(lines) and lines[i+1].strip().startswith("---"):
                        start_idx = i + 2
                    break

        # Read items
        # id quantity length width height weight
        global_id_counter = 0
        for line in lines[start_idx:]:
            if not line.strip(): continue
            parts = line.split()
            if len(parts) >= 6:
                try:
                    # type_id = int(parts[0])
                    quantity = int(parts[1])
                    l = int(parts[2])
                    w = int(parts[3])
                    h = int(parts[4])
                    weight = float(parts[5])
                    
                    for _ in range(quantity):
                        # Create individual items
                        items.append(Item(global_id_counter, l, w, h, weight))
                        global_id_counter += 1
                except ValueError:
                    continue
                    
        return items, bin_dims, max_weight
