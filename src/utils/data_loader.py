from src.domain.item import Item
import random

def load_dataset(path):
    """
    Giả lập load dữ liệu nếu file không tồn tại.
    Trả về list các object Item.
    """
    # TODO: Implement parser for Martello/Berkey Wang format
    # Mock data for demonstration: 50 items with random sizes
    items = []
    for i in range(50):
        # Random dimensions between 10 and 50
        d, w, h = random.randint(10, 50), random.randint(10, 50), random.randint(10, 50)
        items.append(Item(i, (d, w, h)))
    return items