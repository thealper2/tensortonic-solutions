def iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0

    inter_area = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0

def nms(boxes: list, scores: list, iou_threshold: float) -> list:
    """
    Returns a list of retained indices.
    """
    if not boxes:
        return []

    sorted_indices = sorted(range(len(scores)), key=lambda i: (-scores[i], i))

    selected = []

    while sorted_indices:
        current = sorted_indices.pop(0)
        selected.append(current)

        remaining = []
        for idx in sorted_indices:
            if iou(boxes[current], boxes[idx]) < iou_threshold:
                remaining.append(idx)

        sorted_indices = remaining

    return selected