class Item:
    def __init__(self, item_id, dims):
        self.id = item_id
        self.dims = dims  # (Depth, Width, Height)
        self.volume = dims[0] * dims[1] * dims[2]

    def get_dimension(self, orientation):
        """
        Trả về kích thước vật thể dựa trên hướng xoay (0-5).
        Trong 3D có 6 cách xoay khả dĩ.
        [cite: 1161] "It is also assumed that the boxes can be rotated."
        """
        d, w, h = self.dims
        rotations = [
            (d, w, h), (d, h, w),
            (w, d, h), (w, h, d),
            (h, d, w), (h, w, d)
        ]
        return rotations[orientation % 6]