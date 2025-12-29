class ConstraintChecker:
    @staticmethod
    def check_weight(bin_obj, item):
        return bin_obj.current_weight + item.weight <= bin_obj.max_weight

    @staticmethod
    def check_geometry(bin_obj, pos, item_dims):
        """
        Check boundaries and overlaps.
        """
        x, y, z = pos
        d, w, h = item_dims
        bd, bw, bh = bin_obj.dims
        
        # Boundary Check
        if x + d > bd or y + w > bw or z + h > bh:
            return False
            
        # Intersection Check
        for exist_item, exist_pos, exist_dim in bin_obj.items:
            ex, ey, ez = exist_pos
            ed, ew, eh = exist_dim
            
            # Simple AABB intersection
            if (x < ex + ed and x + d > ex and 
                y < ey + ew and y + w > ey and 
                z < ez + eh and z + h > ez):
                return False
                
        return True
    
    @staticmethod
    def can_fit(bin_obj, item, pos, item_dims):
        if not ConstraintChecker.check_weight(bin_obj, item):
            return False
        if not ConstraintChecker.check_geometry(bin_obj, pos, item_dims):
            return False
        return True
