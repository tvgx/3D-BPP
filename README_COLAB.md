# Hướng dẫn chạy Project trên Google Colab

## 1. Chuẩn bị
Tải toàn bộ thư mục project này lên Google Drive hoặc nén thành file `.zip` để upload lên Colab.

## 2. Thiết lập môi trường trên Colab
Tạo một Notebook mới và chạy các lệnh sau:

### Bước 1: Mount Google Drive (Nếu data để trên Drive)
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/My Drive/path/to/project_folder') # Thay đường dẫn thực tế
```

### Bước 2: Upload file (Nếu upload trực tiếp)
Nếu bạn upload file `colab_project.zip` lên Colab:
```python
!unzip colab_project.zip
%cd colab_3d_bpp
```

## 3. Chạy chương trình

### Cách 1: Chạy với một file dữ liệu cụ thể
```bash
!python main.py --algo ga --data data/instances/test-data/Input/3dBPP_1.txt --gen 100
```
Kết quả giải pháp sẽ được in ra màn hình (Terminal) theo format yêu cầu.

### Cách 2: Chạy Batch (Hàng loạt) cho cả thư mục
Chỉ định đường dẫn đến thư mục chứa các file input (hỗ trợ `.txt` format Elhedhli và `.pkl`):
```bash
!python main.py --algo ga --data data/instances/test-data/Input --gen 100
```
Chương trình sẽ tự động duyệt qua tất cả các file trong thư mục và in kết quả lần lượt.

### Cách 3: Chạy với dữ liệu sinh ngẫu nhiên
```bash
!python main.py --algo ga --gen 100
```
Nếu không chỉ định `--data`, chương trình sẽ dùng `dataset.pkl` mặc định hoặc sinh mới.

## 4. Xem kết quả
- **Terminal Output**: Chi tiết các kiện hàng được xếp (Coordinates, Orientation, Bin ID).
- **Biểu đồ hội tụ**: File `convergence_plot.png` sẽ được lưu trong thư mục hiện tại. Bạn có thể xem bằng lệnh:
```python
from IPython.display import Image
Image('convergence_plot.png')
```
