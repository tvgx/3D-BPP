def difference_process(bin_obj, new_item_rect):
    """
    Thực hiện quy trình Difference Process (DP) để cập nhật danh sách khoảng trống.
    Khi một vật được đặt vào, nó sẽ cắt các khoảng trống hiện tại thành các khoảng trống nhỏ hơn.
    Tham khảo: Lai and Chan (1997) cited in[cite: 1212].
    """
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
    """
    Chọn khoảng trống tốt nhất dựa trên heuristic (BBL hoặc BLB).
     "Back-Bottom-Left (BBL) and Back-Left-Bottom (BLB)"
    """
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