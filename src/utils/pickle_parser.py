import pandas as pd
from src.evaluation.packing_simulator import Item

class PickleParser:
    """
    Parser for Pickle (.pkl) datasets containing Pandas DataFrames.
    Expected columns: width, depth, height, weight (optional), volume (optional).
    """
    
    @staticmethod
    def parse(data_path):
        """
        Parses a .pkl file.
        Returns:
            items: List of Item objects.
            bin_dims: (0, 0, 0) as bin dimensions are usually not in the item file.
        """
        items = []
        try:
            df = pd.read_pickle(data_path)
            
            # Ensure it's a DataFrame
            if not isinstance(df, pd.DataFrame):
                print(f"Error: {data_path} does not contain a Pandas DataFrame.")
                return [], (0, 0, 0)
            
            # Map columns to dimensions
            # Data usually has: width, depth, height
            # We map to (D, W, H) or similar.
            # Let's assume standard: Width, Depth, Height.
            # Item(dims=(Depth, Width, Height)) convention in our system? 
            # Our Item takes (D, W, H).
            
            required_cols = ['depth', 'width', 'height']
            for col in required_cols:
                if col not in df.columns:
                    # Try fallback (e.g. Length instead of Depth)
                    # For now strict check
                    print(f"Error: Column '{col}' missing in DataFrame.")
                    print(f"Available columns: {df.columns.tolist()}")
                    return [], (0, 0, 0)

            for idx, row in df.iterrows():
                # ID is the index or mapped from row
                item_id = idx
                
                d = int(row['depth'])
                w = int(row['width'])
                h = int(row['height'])
                
                # Default: All rotations allowed unless specified (usually not in these simple datasets)
                # valid_rotations = list(range(6))
                
                # Check for rotation constraints if columns exist?
                # For now assume free rotation for this dataset type
                
                items.append(Item(item_id, (d, w, h)))
                
        except Exception as e:
            print(f"Error parsing pickle file {data_path}: {e}")
            return [], (0, 0, 0)
            
        return items, (0, 0, 0)
