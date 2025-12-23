import os
from pathlib import Path

# Tên dự án gốc
PROJECT_ROOT = "3d-bpp-evolutionary"

# Định nghĩa nội dung cho từng file trong dự án
# Nội dung được thiết kế dựa trên các tài liệu:
# - BRKGA: brkga-binpacking.pdf [cite: 1148]
# - DE: 7._de_.9m.pdf [cite: 587]
# - PSO: 8._pso_1.7m.pdf [cite: 9]
# - GA: 2._genetic_algorithm-sv_1.2m.pdf [cite: 2664]

file_contents = {
    # ---------------------------------------------------------
    # 1. CONFIGURATION
    # ---------------------------------------------------------
    "config/config.yaml": """
problem:
  # Kích thước thùng chuẩn (ví dụ theo dataset Martello et al.) [cite: 1161]
  bin_dimensions: [100, 100, 100]  # Depth (D), Width (W), Height (H)
  dataset_path: "data/class1_sample.txt"

encoding:
  type: "random_key"
  # Nhiễm sắc thể 3 phần: [Thứ tự gói (BPS) | Heuristic (VPH) | Hướng xoay (VBO)]
  # Tham khảo: Figure 3 - Solution encoding [cite: 1287]
  chromosome_multiplier: 3 

algorithm:
  name: "brkga" # Options: brkga, de, pso, ga
  generations: 300
  # Kích thước quần thể = 30 * số lượng vật thể [cite: 1519]
  pop_size_multiplier: 30
  
  # Tham số BRKGA [cite: 1524]
  brkga:
    elite_pct: 0.10       # Top 10% ưu tú
    mutant_pct: 0.15      # 15% đột biến (ngẫu nhiên mới)
    inheritance_prob: 0.70 # Xác suất thừa kế gen từ cha mẹ ưu tú

  # Tham số Differential Evolution (DE) [cite: 687]
  de:
    crossover_rate: 0.9   # CR
    differential_weight: 0.8 # F

  # Tham số PSO [cite: 71]
  pso:
    w: 0.7    # Quán tính
    c1: 1.5   # Nhận thức (Pbest)
    c2: 1.5   # Xã hội (Gbest)
""",

    # ---------------------------------------------------------
    # 2. MAIN EXECUTION
    # ---------------------------------------------------------
    "main.py": """
import yaml
import numpy as np
from src.utils.data_loader import load_dataset
from src.decoder.decoder import Decoder
from src.algorithms.brkga import BRKGAAlgorithm
from src.algorithms.de import DEAlgorithm
from src.algorithms.pso import PSOAlgorithm
# from src.algorithms.ga import GAAlgorithm

def main():
    print("=== 3D Bin Packing Problem - Unified Evolutionary Framework ===")
    
    # 1. Load Config
    try:
        with open("config/config.yaml", "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: config.yaml not found. Please run the generator script properly.")
        return

    # 2. Load Data (Giả lập nếu không có file thực)
    # Trong thực tế, load từ data/class1.txt
    items = load_dataset(config['problem']['dataset_path'])
    bin_dims = tuple(config['problem']['bin_dimensions'])
    
    print(f"Loaded {len(items)} items.")
    print(f"Bin Dimensions: {bin_dims}")

    # 3. Initialize Decoder (The Bridge)
    # Decoder chứa logic 'Placement Procedure' và 'Maximal Space' [cite: 1210]
    decoder = Decoder(items, bin_dims)
    
    # 4. Select Strategy
    algo_name = config['algorithm']['name']
    print(f"Running Algorithm: {algo_name.upper()}")
    
    solver = None
    if algo_name == 'brkga':
        solver = BRKGAAlgorithm(decoder, config)
    elif algo_name == 'de':
        solver = DEAlgorithm(decoder, config)
    elif algo_name == 'pso':
        solver = PSOAlgorithm(decoder, config)
    else:
        print(f"Algorithm {algo_name} not implemented yet.")
        return
    
    # 5. Run Optimization
    best_solution, best_fitness = solver.solve()
    
    # 6. Decode Final Result
    final_bins = decoder.decode(best_solution)
    print(f"\\nOptimization Complete.")
    print(f"Total Bins Used: {len(final_bins)}")
    print(f"Best Fitness (aNB): {best_fitness:.4f}")

if __name__ == "__main__":
    main()
""",

    "requirements.txt": """
numpy
pyyaml
matplotlib
""",

    # ---------------------------------------------------------
    # 3. UTILS
    # ---------------------------------------------------------
    "src/__init__.py": "",
    "src/utils/__init__.py": "",
    
    "src/utils/data_loader.py": """
from src.domain.item import Item
import random

def load_dataset(path):
    \"\"\"
    Giả lập load dữ liệu nếu file không tồn tại.
    Trả về list các object Item.
    \"\"\"
    # TODO: Implement parser for Martello/Berkey Wang format
    # Mock data for demonstration: 50 items with random sizes
    items = []
    for i in range(50):
        # Random dimensions between 10 and 50
        d, w, h = random.randint(10, 50), random.randint(10, 50), random.randint(10, 50)
        items.append(Item(i, (d, w, h)))
    return items
""",

    # ---------------------------------------------------------
    # 4. DOMAIN LAYER (Physical Objects)
    # ---------------------------------------------------------
    "src/domain/__init__.py": "",

    "src/domain/item.py": """
class Item:
    def __init__(self, item_id, dims):
        self.id = item_id
        self.dims = dims  # (Depth, Width, Height)
        self.volume = dims[0] * dims[1] * dims[2]

    def get_dimension(self, orientation):
        \"\"\"
        Trả về kích thước vật thể dựa trên hướng xoay (0-5).
        Trong 3D có 6 cách xoay khả dĩ.
        [cite: 1161] "It is also assumed that the boxes can be rotated."
        \"\"\"
        d, w, h = self.dims
        rotations = [
            (d, w, h), (d, h, w),
            (w, d, h), (w, h, d),
            (h, d, w), (h, w, d)
        ]
        return rotations[orientation % 6]
""",

    "src/domain/space.py": """
class Space:
    \"\"\"
    Đại diện cho một Maximal Empty Space (Khoảng trống cực đại).
    Được quản lý bởi thuật toán Difference Process.
    [cite: 1213] "A 2D or 3D empty space is maximal if it is not contained in any other space"
    \"\"\"
    def __init__(self, min_coord, max_coord):
        self.min_coord = min_coord # (x, y, z)
        self.max_coord = max_coord # (X, Y, Z)
        self.dims = tuple(M - m for M, m in zip(max_coord, min_coord))
        self.volume = self.dims[0] * self.dims[1] * self.dims[2]
""",

    "src/domain/bin.py": """
from src.domain.space import Space

class Bin:
    def __init__(self, dims):
        self.dims = dims # (D, W, H)
        self.capacity = dims[0] * dims[1] * dims[2]
        self.items = [] # List các vật đã xếp vào
        # Ban đầu, thùng có 1 khoảng trống cực đại bằng kích thước thùng
        self.free_spaces = [Space((0,0,0), dims)]
    
    def get_fill_rate(self):
        used_vol = sum([item.volume for item, _, _ in self.items])
        return used_vol / self.capacity
""",

    # ---------------------------------------------------------
    # 5. DECODER LAYER (Logic Core)
    # ---------------------------------------------------------
    "src/decoder/__init__.py": "",

    "src/decoder/placement.py": """
def difference_process(bin_obj, new_item_rect):
    \"\"\"
    Thực hiện quy trình Difference Process (DP) để cập nhật danh sách khoảng trống.
    Khi một vật được đặt vào, nó sẽ cắt các khoảng trống hiện tại thành các khoảng trống nhỏ hơn.
    Tham khảo: Lai and Chan (1997) cited in[cite: 1212].
    \"\"\"
    new_spaces = []
    # Logic giả định (Cần implement chi tiết thuật toán hình học giao cắt 3D)
    # 1. Loop qua tất cả space trong bin_obj.free_spaces
    # 2. Nếu new_item_rect giao cắt với space -> Phân chia space thành tối đa 6 space con
    # 3. Nếu không giao cắt -> Giữ nguyên space
    # 4. Loại bỏ các space bị chứa hoàn toàn trong space khác (Non-maximal) [cite: 1369]
    
    # Placeholder đơn giản hóa để code chạy được khung:
    # Trong thực tế cần code hình học chi tiết.
    pass 

def select_best_space(bin_obj, item_dims, heuristic_type):
    \"\"\"
    Chọn khoảng trống tốt nhất dựa trên heuristic (BBL hoặc BLB).
     "Back-Bottom-Left (BBL) and Back-Left-Bottom (BLB)"
    \"\"\"
    candidates = []
    for space in bin_obj.free_spaces:
        if (space.dims[0] >= item_dims[0] and 
            space.dims[1] >= item_dims[1] and 
            space.dims[2] >= item_dims[2]):
            candidates.append(space)
    
    if not candidates:
        return None

    # Sắp xếp candidates dựa trên heuristic
    # BBL: Ưu tiên Z (Bottom), sau đó X (Back), sau đó Y (Left) [cite: 1377]
    if heuristic_type == 'BBL':
        candidates.sort(key=lambda s: (s.min_coord[2], s.min_coord[0], s.min_coord[1]))
    # BLB: Ưu tiên Z (Bottom), sau đó Y (Left), sau đó X (Back) [cite: 1380]
    elif heuristic_type == 'BLB':
        candidates.sort(key=lambda s: (s.min_coord[2], s.min_coord[1], s.min_coord[0]))
    
    return candidates[0]
""",

    "src/decoder/decoder.py": """
import numpy as np
from src.domain.bin import Bin
from src.decoder.placement import select_best_space, difference_process

class Decoder:
    \"\"\"
    Lớp cầu nối chuyển đổi Genotype (Vector số thực) -> Phenotype (Danh sách thùng hàng).
    Sử dụng kỹ thuật Random Key Encoding.
    \"\"\"
    def __init__(self, items, bin_dims):
        self.items = items
        self.bin_dims = bin_dims

    def decode(self, chromosome):
        \"\"\"
        Giải mã nhiễm sắc thể 3 phần theo tài liệu BRKGA[cite: 1285].
        Input: Chromosome độ dài 3*N (values 0.0 -> 1.0)
        \"\"\"
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
        \"\"\"
        Tính toán hàm mục tiêu Adjusted Number of Bins (aNB).
        aNB = Số lượng thùng + (Tải trọng thùng ít nhất / Dung tích thùng)
        Mục tiêu: Càng nhỏ càng tốt.
        
        \"\"\"
        bins = self.decode(chromosome)
        num_bins = len(bins)
        if num_bins == 0: return float('inf')
        
        # Tính độ lấp đầy của từng thùng
        loads = [b.get_fill_rate() for b in bins]
        least_load = min(loads) if loads else 0
        
        return num_bins + least_load
""",

    # ---------------------------------------------------------
    # 6. ALGORITHMS LAYER (Optimization Solvers)
    # ---------------------------------------------------------
    "src/algorithms/__init__.py": "",

    "src/algorithms/base_solver.py": """
import numpy as np

class BaseSolver:
    def __init__(self, decoder, config):
        self.decoder = decoder
        self.config = config
        self.items_count = len(decoder.items)
        
        # Độ dài nhiễm sắc thể = 3 * N [cite: 1285]
        self.chromosome_len = 3 * self.items_count
        
        # Khởi tạo quần thể ngẫu nhiên [0, 1]
        self.pop_size = config['algorithm']['pop_size_multiplier'] * self.items_count
        self.population = np.random.rand(self.pop_size, self.chromosome_len)
        self.fitnesses = np.array([float('inf')] * self.pop_size)
        
        self.best_solution = None
        self.best_fitness = float('inf')

    def evaluate_all(self):
        \"\"\"Đánh giá fitness cho toàn bộ quần thể\"\"\"
        # Trong thực tế nên dùng Parallel processing [cite: 1405]
        for i in range(self.pop_size):
            fit = self.decoder.get_fitness(self.population[i])
            self.fitnesses[i] = fit
            
            if fit < self.best_fitness:
                self.best_fitness = fit
                self.best_solution = self.population[i].copy()
""",

    "src/algorithms/brkga.py": """
import numpy as np
from src.algorithms.base_solver import BaseSolver

class BRKGAAlgorithm(BaseSolver):
    \"\"\"
    Biased Random-Key Genetic Algorithm.
    Dựa trên tài liệu [cite: 1148] và[cite: 1207].
    \"\"\"
    def solve(self):
        params = self.config['algorithm']['brkga']
        num_elite = int(params['elite_pct'] * self.pop_size)
        num_mutant = int(params['mutant_pct'] * self.pop_size)
        num_crossover = self.pop_size - num_elite - num_mutant
        rho = params['inheritance_prob'] # Xác suất lấy gen từ Elite [cite: 1271]

        for gen in range(self.config['algorithm']['generations']):
            # 1. Đánh giá và Sắp xếp quần thể
            self.evaluate_all()
            sorted_indices = np.argsort(self.fitnesses)
            sorted_pop = self.population[sorted_indices]
            
            print(f"Gen {gen}: Best Fitness aNB = {self.best_fitness:.4f}")

            # 2. Phân loại Elite và Non-Elite [cite: 1260]
            elites = sorted_pop[:num_elite]
            non_elites = sorted_pop[num_elite:]

            # 3. Tạo thế hệ mới
            new_pop = []
            
            # 3.1. Copy Elites (Elitism)
            new_pop.extend(elites)
            
            # 3.2. Generate Mutants (Ngẫu nhiên hoàn toàn) [cite: 1263]
            mutants = np.random.rand(num_mutant, self.chromosome_len)
            new_pop.extend(mutants)
            
            # 3.3. Crossover (Biased) [cite: 1267]
            # Lai ghép giữa 1 Elite và 1 Non-Elite
            for _ in range(num_crossover):
                elite_parent = elites[np.random.randint(0, num_elite)]
                non_elite_parent = non_elites[np.random.randint(0, len(non_elites))]
                
                # Parameterized Uniform Crossover
                offspring = np.where(np.random.rand(self.chromosome_len) < rho,
                                     elite_parent,
                                     non_elite_parent)
                new_pop.append(offspring)
            
            self.population = np.array(new_pop)
            
        return self.best_solution, self.best_fitness
""",

    "src/algorithms/de.py": """
import numpy as np
from src.algorithms.base_solver import BaseSolver

class DEAlgorithm(BaseSolver):
    \"\"\"
    Differential Evolution (DE).
    Dựa trên tài liệu[cite: 587].
    Phù hợp với Random Key Encoding vì hoạt động trên số thực.
    \"\"\"
    def solve(self):
        params = self.config['algorithm']['de']
        F = params['differential_weight'] # [cite: 689]
        CR = params['crossover_rate']     # [cite: 688]

        for gen in range(self.config['algorithm']['generations']):
            self.evaluate_all()
            print(f"Gen {gen}: Best Fitness aNB = {self.best_fitness:.4f}")
            
            new_pop = np.copy(self.population)
            
            for i in range(self.pop_size):
                # 1. Mutation: Chọn r1, r2, r3 khác i [cite: 650]
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                r1, r2, r3 = self.population[np.random.choice(idxs, 3, replace=False)]
                
                # Vector đột biến V = r1 + F * (r2 - r3) [cite: 652]
                mutant_vector = r1 + F * (r2 - r3)
                # Đảm bảo giá trị nằm trong [0, 1] (Random Key constraint)
                mutant_vector = np.clip(mutant_vector, 0, 1)
                
                # 2. Crossover (Binomial) [cite: 662]
                cross_points = np.random.rand(self.chromosome_len) < CR
                # Đảm bảo ít nhất 1 gen được thay đổi
                k = np.random.randint(0, self.chromosome_len)
                cross_points[k] = True
                
                trial_vector = np.where(cross_points, mutant_vector, self.population[i])
                
                # 3. Selection [cite: 668]
                # DE đánh giá ngay lập tức (greedy selection)
                f_trial = self.decoder.get_fitness(trial_vector)
                if f_trial <= self.fitnesses[i]: # Minimize problem
                    new_pop[i] = trial_vector
            
            self.population = new_pop

        return self.best_solution, self.best_fitness
""",

    "src/algorithms/pso.py": """
import numpy as np
from src.algorithms.base_solver import BaseSolver

class PSOAlgorithm(BaseSolver):
    \"\"\"
    Particle Swarm Optimization (PSO).
    Dựa trên tài liệu[cite: 9].
    Particles bay trong không gian hypercube [0, 1]^3N.
    \"\"\"
    def solve(self):
        params = self.config['algorithm']['pso']
        w = params['w']   # Quán tính [cite: 47]
        c1 = params['c1'] # Nhận thức
        c2 = params['c2'] # Xã hội

        # Khởi tạo vận tốc
        velocities = np.random.rand(self.pop_size, self.chromosome_len) * 0.1
        
        # Pbest (Cá thể tốt nhất của từng hạt)
        pbest_pos = np.copy(self.population)
        pbest_val = np.array([float('inf')] * self.pop_size)
        
        # Gbest (Tốt nhất toàn đàn)
        gbest_pos = None
        gbest_val = float('inf')

        for gen in range(self.config['algorithm']['generations']):
            # Đánh giá fitness
            for i in range(self.pop_size):
                fit = self.decoder.get_fitness(self.population[i])
                
                # Cập nhật Pbest [cite: 26]
                if fit < pbest_val[i]:
                    pbest_val[i] = fit
                    pbest_pos[i] = self.population[i].copy()
                
                # Cập nhật Gbest [cite: 29]
                if fit < gbest_val:
                    gbest_val = fit
                    gbest_pos = self.population[i].copy()
            
            print(f"Gen {gen}: Best Fitness aNB = {gbest_val:.4f}")
            
            # Cập nhật Vận tốc và Vị trí [cite: 46]
            r1 = np.random.rand(self.pop_size, self.chromosome_len)
            r2 = np.random.rand(self.pop_size, self.chromosome_len)
            
            velocities = (w * velocities + 
                          c1 * r1 * (pbest_pos - self.population) + 
                          c2 * r2 * (gbest_pos - self.population))
            
            self.population = self.population + velocities
            
            # Quan trọng: Kẹp giá trị về [0, 1] để giữ đúng tính chất Random Key
            self.population = np.clip(self.population, 0, 1)

        return gbest_pos, gbest_val
""",
}

def create_project_structure():
    """Tạo thư mục và ghi file"""
    print(f"Creating project: {PROJECT_ROOT}...")
    
    for data_path, content in file_contents.items():
        # Xây dựng đường dẫn đầy đủ
        full_path = Path(PROJECT_ROOT) / data_path
        
        # Tạo thư mục cha nếu chưa tồn tại
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ghi nội dung vào file
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content.strip())
        
        print(f"Created: {full_path}")

    print("\\nDone! Project structure generated successfully.")
    print(f"Run 'cd {PROJECT_ROOT} && python main.py' to start.")

if __name__ == "__main__":
    create_project_structure()
