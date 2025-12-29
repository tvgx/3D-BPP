class Item:
    def __init__(self, id, d, w, h, weight, type_id=0):
        self.id = id
        self.dims = (d, w, h) # L, W, H
        self.weight = weight
        self.type_id = type_id # User requested type field
    
    def __repr__(self):
        return f"Item({self.id}, dims={self.dims}, weight={self.weight})"

class Bin:
    def __init__(self, dims, max_weight):
        self.dims = dims # (D, W, H)
        self.max_weight = max_weight
        self.current_weight = 0
        self.items = [] # List of (item, position, rotation_type, packed_dims)
        
    def add_item(self, item, pos, dims_rot):
        self.items.append((item, pos, dims_rot))
        self.current_weight += item.weight
