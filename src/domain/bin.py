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