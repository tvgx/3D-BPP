# Hướng dẫn chạy các thuật toán (DE, ES, GA)

Hệ thống hiện tại hỗ trợ các thuật toán: **Differential Evolution (DE)**, **Evolution Strategies (ES/CMA-ES)**, và **Genetic Algorithm (GA)**.
Các thuật toán cũ (ACO, PSO) đã được loại bỏ theo yêu cầu.

## 1. Cấu trúc lệnh cơ bản

Chạy chương trình thông qua `main.py`:

```bash
python main.py --algo [ALGO_NAME] --data [PATH_TO_DATA]
```

- `--algo`: Tên thuật toán (`de`, `es`, `ga`, `brkga`, `mfea`, `nsga2`).
- `--data`: Đường dẫn đến file dữ liệu (hỗ trợ `.txt` mendeley hoặc `.pkl`).
- `--config`: (Tùy chọn) Đường dẫn file cấu hình tùy chỉnh để ghi đè mặc định.
- `--viz`: (Tùy chọn) Tạo file trực quan hóa HTML sau khi chạy.

## 2. Hướng dẫn chi tiết từng thuật toán

### 2.1. Differential Evolution (DE)
Thuật toán sử dụng biến thể **L-SHADE** (Success-History based Adaptive Differential Evolution).

**Lệnh chạy:**
```bash
python main.py --algo de --data data/instances/mendeley_v2/sample.txt
```

**Cấu hình (`config/de_shade.yaml`):**
- `pop_size`: Kích thước quần thể.
- `memory_size`: Kích thước bộ nhớ lịch sử cho tham số F và Cr.

### 2.2. Evolution Strategies (ES)
Thuật toán sử dụng **CMA-ES** (Covariance Matrix Adaptation Evolution Strategy). Rất mạnh cho các bài toán liên tục.

**Lệnh chạy:**
```bash
python main.py --algo es --data data/instances/mendeley_v2/sample.txt
```

**Cấu hình (`config/es_cma_es.yaml`):**
- `pop_size`: Kích thước quần thể (Mặc định tự động tính theo `4 + 3*log(N)`).
- `sigma`: Bước nhảy khởi tạo (Step size).

### 2.3. Genetic Algorithm (GA)
Thuật toán Di truyền cổ điển với biểu diễn **Random Key**.
- **Selection**: Tournament Selection.
- **Crossover**: Uniform Crossover.
- **Mutation**: Gaussian Mutation.

**Lệnh chạy:**
```bash
python main.py --algo ga --data data/instances/mendeley_v2/sample.txt
```

**Cấu hình (`config/ga_classic.yaml`):**
- `population_size_multiplier`: Hệ số nhân kích thước quần thể (Pop = N * multiplier).
- `crossover_prob`: Xác suất lai ghép (Default: 0.9).
- `mutation_prob_factor`: Hệ số điều chỉnh xác suất đột biến.

## 3. Ví dụ chạy với trực quan hóa
Để xem kết quả xếp thùng 3D:

```bash
python main.py --algo ga --viz
```
File kết quả sẽ được lưu tại `result_visualization.html`.
