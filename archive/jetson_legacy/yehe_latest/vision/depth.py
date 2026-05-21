import numpy as np
import pyzed.sl as sl

def get_zed_depth_distance(depth_map, x1, y1, x2, y2):

    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)

    err, depth_value = depth_map.get_value(center_x, center_y)

    if err != sl.ERROR_CODE.SUCCESS:
        return None

    if np.isnan(depth_value) or np.isinf(depth_value):
        return None

    if depth_value <= 0:
        return None

    if depth_value > 10:
        return None

    return float(depth_value)