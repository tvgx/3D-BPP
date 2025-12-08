import numpy as np
from src.domain.bin import Bin
from src.decoder.placement import select_best_space, difference_process

class Decoder:
    """
    Lớp cầu nối chuyển đổi Genotype (Vector số thực) -> Phenotype (Danh sách thùng hàng).
    Sử dụng kỹ thuật Random Key Encoding.
    """
    def __init__(self, items, bin_dims):
        self.items = items
        self.bin_dims = bin_dims

    def decode(self, chromosome):
        """
        Giải mã nhiễm sắc thể 3 phần theo tài liệu BRKGA[cite: 1285].
        Input: Chromosome độ dài 3*N (values 0.0 -> 1.0)
        """
        n = len(self.items)
        
        # --- Phần 1: Box Packing Sequence (BPS) ---
        # Sắp xếp items dựa trên giá trị gene tăng dần 
        bps_keys = chromosome[:n]
        sorted_indices = np.argsort(bps_keys)
        
        # --- Phần 2: Vector of Placement Heuristics (VPH) ---
        # Gene <= 0.5 -> BBL, Gene > 0.5 -> BLB [cite: 1331]
        vph_keys = chromosome[n:2*n]
        heuristics = ['BBL' if k <= 0.5 else 'BLB' for k in vph_keys]
        
        # --- Phần 3: Vector of Box Orientations (VBO) ---
        # Chọn 1 trong 6 hướng xoay [cite: 1334]
        vbo_keys = chromosome[2*n:]
        # Công thức: ceil(gene * number_of_rotations)
        rotations = [int(np.ceil(k * 6)) - 1 for k in vbo_keys] # map to 0-5

        bins = []
        # Mở thùng đầu tiên
        bins.append(Bin(self.bin_dims))

        # --- Placement Procedure [cite: 1383] ---
        for i in range(n):
            item_idx = sorted_indices[i]
            item = self.items[item_idx]
            heuristic = heuristics[item_idx]
            rotation_idx = rotations[item_idx]
            
            # Lấy kích thước thực tế sau khi xoay
            current_dims = item.get_dimension(rotation_idx)
            
            placed = False
            # Thử đặt vào các thùng hiện có (First Fit rule) [cite: 1396]
            for b in bins:
                best_space = select_best_space(b, current_dims, heuristic)
                if best_space:
                    # Đặt vật vào (logic đơn giản hóa cho demo)
                    # Trong thực tế phải gọi difference_process để cắt space
                    b.items.append((item, best_space.min_coord, current_dims))
                    # difference_process(b, ...) 
                    placed = True
                    break
            
            # Nếu không thùng nào vừa, mở thùng mới
            if not placed:
                new_bin = Bin(self.bin_dims)
                # Đặt vào gốc tọa độ (0,0,0) [cite: 1361]
                new_bin.items.append((item, (0,0,0), current_dims))
                bins.append(new_bin)
        
        return bins

    def get_fitness(self, chromosome):
        """
        Tính toán hàm mục tiêu Adjusted Number of Bins (aNB).
        aNB = Số lượng thùng + (Tải trọng thùng ít nhất / Dung tích thùng)
        Mục tiêu: Càng nhỏ càng tốt.
        
        """
        bins = self.decode(chromosome)
        num_bins = len(bins)
        if num_bins == 0: return float('inf')
        
        # Tính độ lấp đầy của từng thùng
        loads = [b.get_fill_rate() for b in bins]
        least_load = min(loads) if loads else 0
        
        return num_bins + least_load