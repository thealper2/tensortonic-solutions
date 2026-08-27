import math

def roi_pool(feature_map: list, rois: list, output_size: int) -> list:
    """
    Returns a list of pooled grids.
    """
    H = len(feature_map)
    W = len(feature_map[0]) if H > 0 else 0
    S = output_size

    result = []

    for roi in rois:
        x1, y1, x2, y2 = roi
        H_R = y2 - y1
        W_R = x2 - x1

        pooled = [[0] * S for _ in range(S)]

        for i in range(S):
            for j in range(S):
                row_start = y1 + math.floor(i * H_R / S)
                row_end = y1 + math.floor((i + 1) * H_R / S)
                col_start = x1 + math.floor(j * W_R / S)
                col_end = x1 + math.floor((j + 1) * W_R / S)

                if row_end <= row_start:
                    row_end = row_start + 1
                if col_end <= col_start:
                    col_end = col_start + 1

                max_val = float('-inf')
                for r in range(row_start, row_end):
                    for c in range(col_start, col_end):
                        val = feature_map[r][c]
                        if val > max_val:
                            max_val = val

                pooled[i][j] = max_val

        result.append(pooled)

    return result