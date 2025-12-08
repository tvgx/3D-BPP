class Space:
    """
    Đại diện cho một Maximal Empty Space (Khoảng trống cực đại).
    Được quản lý bởi thuật toán Difference Process.
    [cite: 1213] "A 2D or 3D empty space is maximal if it is not contained in any other space"
    """
    def __init__(self, min_coord, max_coord):
        self.min_coord = min_coord # (x, y, z)
        self.max_coord = max_coord # (X, Y, Z)
        self.dims = tuple(M - m for M, m in zip(max_coord, min_coord))
        self.volume = self.dims[0] * self.dims[1] * self.dims[2]