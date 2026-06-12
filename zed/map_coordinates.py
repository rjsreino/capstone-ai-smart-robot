import math
import os


MAP_CELL_SIZE_M = float(os.getenv("VICKY_MAP_CELL_SIZE_M", "0.1"))
MAP_WIDTH = int(os.getenv("VICKY_MAP_WIDTH", "100"))
MAP_HEIGHT = int(os.getenv("VICKY_MAP_HEIGHT", "100"))

# Map convention used by the app:
#   grid_x grows to the right/east
#   grid_z grows downward/south
#   yaw 0 faces up/north, yaw 90 faces right/east
POSE_X_SIGN = float(os.getenv("VICKY_ZED_X_SIGN", "1.0"))
POSE_Z_SIGN = float(os.getenv("VICKY_ZED_Z_SIGN", "-1.0"))
POSE_YAW_SIGN = float(os.getenv("VICKY_ZED_YAW_SIGN", "-1.0"))
POSE_YAW_OFFSET_DEG = float(os.getenv("VICKY_ZED_HEADING_OFFSET_DEG", "0.0"))
DISPLAY_YAW_SIGN = float(os.getenv("VICKY_DISPLAY_YAW_SIGN", "-1.0"))
DISPLAY_YAW_OFFSET_DEG = float(os.getenv("VICKY_DISPLAY_YAW_OFFSET_DEG", "0.0"))
PROJECTION_YAW_SIGN = float(os.getenv("VICKY_PROJECTION_YAW_SIGN", "-1.0"))
PROJECTION_YAW_OFFSET_DEG = float(os.getenv("VICKY_PROJECTION_YAW_OFFSET_DEG", "0.0"))
CAMERA_X_SIGN = float(os.getenv("VICKY_CAMERA_X_SIGN", "1.0"))


def clamp(value, low, high):
    return max(low, min(value, high))


def normalize_degrees(value):
    return (float(value) + 180.0) % 360.0 - 180.0


def raw_yaw_to_map_yaw(raw_yaw):
    return normalize_degrees(float(raw_yaw) * POSE_YAW_SIGN + POSE_YAW_OFFSET_DEG)


def base_yaw_to_display_yaw(base_yaw):
    return normalize_degrees(float(base_yaw) * DISPLAY_YAW_SIGN + DISPLAY_YAW_OFFSET_DEG)


def base_yaw_to_projection_yaw(base_yaw):
    return normalize_degrees(float(base_yaw) * PROJECTION_YAW_SIGN + PROJECTION_YAW_OFFSET_DEG)


def pose_mm_to_grid(x_mm, z_mm, cell_size_m=MAP_CELL_SIZE_M, width=MAP_WIDTH, height=MAP_HEIGHT):
    x_m = float(x_mm or 0.0) / 1000.0
    z_m = float(z_mm or 0.0) / 1000.0
    grid_x = int((x_m * POSE_X_SIGN) / cell_size_m) + width // 2
    grid_z = int((z_m * POSE_Z_SIGN) / cell_size_m) + height // 2
    return (
        clamp(grid_z, 0, height - 1),
        clamp(grid_x, 0, width - 1),
    )


def image_x_to_camera_x(depth_m, pixel_x, frame_width, fov_rad):
    if frame_width <= 0:
        return 0.0
    angle_rad = -float(fov_rad) / 2.0 + float(pixel_x) * (float(fov_rad) / float(frame_width))
    return CAMERA_X_SIGN * float(depth_m) * math.sin(angle_rad)


def image_x_to_camera_point(depth_m, pixel_x, frame_width, fov_rad):
    if frame_width <= 0:
        return 0.0, float(depth_m)
    angle_rad = -float(fov_rad) / 2.0 + float(pixel_x) * (float(fov_rad) / float(frame_width))
    x_c = CAMERA_X_SIGN * float(depth_m) * math.sin(angle_rad)
    z_c = float(depth_m) * math.cos(angle_rad)
    return x_c, z_c


def camera_point_to_grid(user_grid_x, user_grid_z, map_yaw_deg, x_c, z_c, cell_size_m=MAP_CELL_SIZE_M, width=MAP_WIDTH, height=MAP_HEIGHT):
    yaw_rad = math.radians(float(map_yaw_deg))
    sin_yaw = math.sin(yaw_rad)
    cos_yaw = math.cos(yaw_rad)

    # Camera z is forward. With yaw 0, forward means grid_z decreases.
    map_dx_m = float(x_c) * cos_yaw + float(z_c) * sin_yaw
    map_dz_m = float(x_c) * sin_yaw - float(z_c) * cos_yaw

    grid_x = int(round(float(user_grid_x) + map_dx_m / cell_size_m))
    grid_z = int(round(float(user_grid_z) + map_dz_m / cell_size_m))
    return (
        clamp(grid_x, 0, width - 1),
        clamp(grid_z, 0, height - 1),
    )


def bearing_to_grid_delta(dx, dz):
    return math.degrees(math.atan2(float(dx), -float(dz)))


def forward_grid_cell(user_grid_z, user_grid_x, map_yaw_deg, step_cells, width=MAP_WIDTH, height=MAP_HEIGHT):
    yaw_rad = math.radians(float(map_yaw_deg))
    x = int(round(float(user_grid_x) + math.sin(yaw_rad) * float(step_cells)))
    z = int(round(float(user_grid_z) - math.cos(yaw_rad) * float(step_cells)))
    return (
        clamp(z, 0, height - 1),
        clamp(x, 0, width - 1),
    )
