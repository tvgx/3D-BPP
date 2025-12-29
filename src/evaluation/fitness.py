class FitnessEvaluator:
    @staticmethod
    def calculate_anb(bins):
        """
        aNB = N + (V_least_loaded / V_bin_capacity)
        Mục tiêu: Giảm thiểu số thùng và dồn hộp để làm trống thùng ít đồ nhất.
        """
        if not bins: return float('inf')
        
        n_bins = len(bins)
        # Tìm thùng có tải trọng thấp nhất để tối ưu hóa việc loại bỏ thùng đó
        volumes = [sum(item.volume for item, _, _ in b.items) for b in bins]
        min_load = min(volumes)
        capacity = bins[0].capacity
        
        return n_bins + (min_load / capacity)