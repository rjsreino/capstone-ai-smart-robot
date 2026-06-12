#!/usr/bin/env python3
"""
ZED Camera Depth Processing Module
Handles stereo depth map generation and processing for navigation.
Compatible with ZED 1 cameras connected via USB 2.0 (720p @ 15fps).
"""

import cv2
import pyzed.sl as sl
import numpy as np
import os
from typing import Tuple, Optional, Dict
import time
from dataclasses import dataclass
from map_coordinates import (
    CAMERA_X_SIGN,
    MAP_CELL_SIZE_M,
    POSE_X_SIGN,
    POSE_YAW_OFFSET_DEG,
    POSE_YAW_SIGN,
    POSE_Z_SIGN,
    base_yaw_to_projection_yaw,
    camera_point_to_grid,
    pose_mm_to_grid,
)


@dataclass
class ZedDepthConfig:
    """Configuration for ZED depth processing"""
    # ZED camera settings
    resolution: str = "vga"  # For ZED 1 USB 2.0 connection, use vga (720p fails with 1893)
    fps: int = 15             # For ZED 1 USB 2.0 connection, use 15 fps
    depth_mode: str = "PERFORMANCE" # ULTRA, QUALITY, PERFORMANCE, NEURAL
    
    # Depth range (in millimeters)
    min_depth: int = 300      # 30cm minimum
    max_depth: int = 10000    # 10m maximum (reduced for indoor navigation)
    invalid_depth_as_far: bool = True  # Treat no-return / beyond-range depth as open space for navigation
    
    # Grid settings for navigation
    grid_width: int = 64      # Divide depth map into grid
    grid_height: int = 48
    wall_hit_threshold: int = int(os.getenv("VICKY_WALL_HIT_THRESHOLD", "3"))
    free_hit_threshold: int = int(os.getenv("VICKY_FREE_HIT_THRESHOLD", "2"))
    wall_confidence_threshold: float = float(os.getenv("VICKY_WALL_CONFIDENCE_THRESHOLD", "0.58"))
    wall_inflation_cells: int = int(os.getenv("VICKY_WALL_INFLATION_CELLS", "1"))
    ray_clear_limit_m: float = float(os.getenv("VICKY_RAY_CLEAR_LIMIT_M", "4.5"))
    map_update_stride: int = int(os.getenv("VICKY_MAP_UPDATE_EVERY_N", "3"))
    enable_obstacle_map: bool = os.getenv("VICKY_ENABLE_OBSTACLE_MAP", "0").strip().lower() in {"1", "true", "yes"}
    cell_size_m: float = MAP_CELL_SIZE_M
    x_sign: float = POSE_X_SIGN
    z_sign: float = POSE_Z_SIGN
    yaw_sign: float = POSE_YAW_SIGN
    heading_offset_deg: float = POSE_YAW_OFFSET_DEG
    camera_x_sign: float = CAMERA_X_SIGN


class ZedDepthProcessor:
    """
    Process depth data from ZED 1 camera for navigation.
    Implements the same API interface as OAK-D DepthProcessor for drop-in compatibility.
    """
    
    def __init__(self, config: Optional[ZedDepthConfig] = None):
        self.config = config or ZedDepthConfig()
        self.zed = None
        
        # Containers for ZED data
        self.image_left = sl.Mat()
        self.depth_map = sl.Mat()
        self.runtime_parameters = sl.RuntimeParameters()
        
        # Statistics
        self.frame_count = 0
        self.fps = 0.0
        self.last_fps_time = time.time()
        
        # Health monitoring
        self.last_frame_time = time.time()
        self.none_frame_count = 0
        self.good_frame_count = 0
        
        # Positional tracking & SLAM data
        self.tx = 0.0
        self.ty = 0.0
        self.tz = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.raw_yaw = 0.0
        self.yaw = 0.0
        self.positional_tracking_enabled = False
        self.is_tracking_ok = False
        self.tracking_state = "DISABLED"
        self.occupancy_grid = np.full((100, 100), 2, dtype=np.int8)
        self.wall_hit_grid = np.zeros((100, 100), dtype=np.float32)
        self.free_hit_grid = np.zeros((100, 100), dtype=np.float32)
        self.nav_frame_count = 0

    def reset_occupancy_grid(self):
        """Reset mapping confidence while preserving the tri-state occupancy convention."""
        self.occupancy_grid.fill(2)
        self.wall_hit_grid.fill(0)
        self.free_hit_grid.fill(0)

    def _mark_free_cell(self, grid_z: int, grid_x: int) -> None:
        if not (0 <= grid_z < 100 and 0 <= grid_x < 100):
            return
        self.free_hit_grid[grid_z, grid_x] = min(self.free_hit_grid[grid_z, grid_x] + 1.0, 30.0)
        self.wall_hit_grid[grid_z, grid_x] = max(self.wall_hit_grid[grid_z, grid_x] - 0.35, 0.0)

    def _mark_wall_cell(self, grid_z: int, grid_x: int) -> None:
        if not (0 <= grid_z < 100 and 0 <= grid_x < 100):
            return
        self.wall_hit_grid[grid_z, grid_x] = min(self.wall_hit_grid[grid_z, grid_x] + 1.0, 30.0)
        self.free_hit_grid[grid_z, grid_x] = max(self.free_hit_grid[grid_z, grid_x] - 0.15, 0.0)

    def _refresh_occupancy_from_confidence(self) -> None:
        total = self.wall_hit_grid + self.free_hit_grid
        wall_confidence = np.divide(
            self.wall_hit_grid,
            np.maximum(total, 1.0),
            out=np.zeros_like(self.wall_hit_grid),
            where=total > 0,
        )
        occupied = (
            (self.wall_hit_grid >= self.config.wall_hit_threshold)
            & (wall_confidence >= self.config.wall_confidence_threshold)
        )
        free = (self.free_hit_grid >= self.config.free_hit_threshold) & ~occupied

        grid = np.full((100, 100), 2, dtype=np.int8)
        grid[free] = 0
        grid[occupied] = 1

        inflation = max(0, int(self.config.wall_inflation_cells))
        if inflation > 0 and np.any(occupied):
            kernel_size = inflation * 2 + 1
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            inflated = cv2.dilate(occupied.astype(np.uint8), kernel, iterations=1) > 0
            grid[inflated] = 1

        self.occupancy_grid = grid
        
    def start(self):
        """Start the ZED camera connection"""
        print("[ZedDepthProcessor] Starting ZED Camera...")
        
        self.zed = sl.Camera()
        
        # Configuration parameters
        init_params = sl.InitParameters()
        
        # 1. RESOLUTION & FPS SETTINGS
        # Map string resolution to pyzed.sl.RESOLUTION
        res_map = {
            "2k": sl.RESOLUTION.HD2K,
            "1080p": sl.RESOLUTION.HD1080,
            "720p": sl.RESOLUTION.HD720,
            "vga": sl.RESOLUTION.VGA
        }
        selected_res = res_map.get(self.config.resolution.lower(), sl.RESOLUTION.HD720)
        init_params.camera_resolution = selected_res
        init_params.camera_fps = self.config.fps
        
        # 2. DEPTH MODE
        # Map depth mode string to pyzed.sl.DEPTH_MODE
        depth_mode_map = {
            "NONE": sl.DEPTH_MODE.NONE,
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "NEURAL": sl.DEPTH_MODE.NEURAL
        }
        init_params.depth_mode = depth_mode_map.get(self.config.depth_mode.upper(), sl.DEPTH_MODE.ULTRA)
        
        # Coordinate units (we use millimeters for consistency with UGV systems)
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        
        # Lock depth range inside SDK
        init_params.depth_minimum_distance = self.config.min_depth
        init_params.depth_maximum_distance = self.config.max_depth
        
        # 3. COMPATIBILITY & HANDSHAKE FIXES FOR ZED 1 & USB 2.0
        # - Disable sensors (IMU) requirements, as original ZED 1 has no hardware sensors
        init_params.sensors_required = False
        
        # Attempt connection with retries
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            print(f"[ZedDepthProcessor] Connection attempt {attempt + 1}/{max_retries}...")
            status = self.zed.open(init_params)
            if status == sl.ERROR_CODE.SUCCESS:
                print("[ZedDepthProcessor] Connection established successfully!")
                break
            else:
                print(f"[ZedDepthProcessor] Failed to open ZED camera (Error Code: {status})")
                if status == sl.ERROR_CODE.CORRUPTED_SDK_INSTALLATION:
                    print("[ZedDepthProcessor] Tip: If using NEURAL depth mode, this error indicates TensorRT is missing/uninstalled.")
                    print("[ZedDepthProcessor]      Please configure depth_mode to 'ULTRA', 'QUALITY', or 'PERFORMANCE' in the settings.")
                
                if attempt < max_retries - 1:
                    print(f"Waiting {retry_delay}s before retrying...")
                    time.sleep(retry_delay)
                else:
                    print("\n--- ZED CONNECTION TROUBLESHOOTING ---")
                    print("1. Ensure the ZED USB cable is plugged in firmly.")
                    print("2. Since you are using a USB 2.0 connection, ONLY VGA resolution is supported (720p or above will fail with Error 1893).")
                    print("3. Ensure no other applications (like ZED Explorer, ZED Depth Viewer, or another Python process) are currently holding the camera open.")
                    print("4. Try replugging the camera into the USB port or a different USB port to reset the connection.")
                    raise RuntimeError(f"Could not connect to ZED camera after {max_retries} attempts: {status}")
        
        # Enable auto exposure and auto gain to fix exposure issues
        self.zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 1)
        
        # Set runtime parameters
        self.runtime_parameters.confidence_threshold = 50
        self.runtime_parameters.texture_confidence_threshold = 100
        
        info = self.zed.get_camera_information()
        fw = info.camera_configuration.firmware_version
        model = info.camera_model
        print(f"[ZedDepthProcessor] Model: {model} | Firmware: {fw}")
        print(f"  Resolution: {info.camera_configuration.resolution.width}x{info.camera_configuration.resolution.height}")
        print(f"  Target FPS: {self.config.fps}")
        print(f"  Depth Mode: {self.config.depth_mode}")
        print(f"  Depth Range: {self.config.min_depth}-{self.config.max_depth}mm")
        
        # Enable Positional Tracking / SLAM only when explicitly requested.
        # ZED 1 can spam keyframe-memory errors during long scans; our saved
        # spatial memory is handled separately in zed/maps, so tracking is
        # optional for the live demo path.
        tracking_requested = os.getenv("VICKY_ZED_TRACKING", "1").strip().lower() in {"1", "true", "yes"}
        if tracking_requested:
            tracking_params = sl.PositionalTrackingParameters()
            if hasattr(tracking_params, "enable_area_memory"):
                tracking_params.enable_area_memory = os.getenv("VICKY_ZED_AREA_MEMORY", "0").strip().lower() in {"1", "true", "yes"}
            if hasattr(tracking_params, "enable_pose_smoothing"):
                tracking_params.enable_pose_smoothing = True
            status_tracking = self.zed.enable_positional_tracking(tracking_params)
            if status_tracking == sl.ERROR_CODE.SUCCESS:
                self.positional_tracking_enabled = True
                self.tracking_state = "OK"
                print("[ZedDepthProcessor] SLAM Positional Tracking enabled successfully.")
            else:
                self.positional_tracking_enabled = False
                self.tracking_state = str(status_tracking)
                print(f"[ZedDepthProcessor WARNING] Failed to enable positional tracking: {status_tracking}")
        else:
            self.positional_tracking_enabled = False
            self.tracking_state = "DISABLED"
            print("[ZedDepthProcessor] ZED positional tracking disabled (set VICKY_ZED_TRACKING=1 to enable).")
        
    def stop(self):
        """Stop ZED camera connection"""
        if self.zed:
            self.zed.close()
            self.zed = None
        print("[ZedDepthProcessor] Stopped")
        
    def restart(self):
        """Restart connection"""
        print("[ZedDepthProcessor] Restarting...")
        self.stop()
        time.sleep(1)
        self.start()
        self.none_frame_count = 0
        self.good_frame_count = 0
        print("[ZedDepthProcessor] Restart complete")
        
    def grab_frame(self) -> bool:
        """Grab a frame from the ZED camera"""
        if not self.zed:
            return False
        
        status = self.zed.grab(self.runtime_parameters)
        if status == sl.ERROR_CODE.SUCCESS:
            self.good_frame_count += 1
            self.last_frame_time = time.time()
            
            # Retrieve camera position relative to World frame
            self.is_tracking_ok = False
            if self.positional_tracking_enabled:
                try:
                    camera_pose = sl.Pose()
                    state = self.zed.get_position(camera_pose, sl.REFERENCE_FRAME.WORLD)
                    self.is_tracking_ok = (state == sl.POSITIONAL_TRACKING_STATE.OK)
                    self.tracking_state = "OK" if self.is_tracking_ok else str(state)
                    if self.is_tracking_ok:
                        translation = camera_pose.get_translation().get()
                        self.tx = float(translation[0])
                        self.ty = float(translation[1])
                        self.tz = float(translation[2])
                        
                        # Euler orientation from orientation quaternion
                        orientation = camera_pose.get_orientation().get()
                        qx, qy, qz, qw = orientation[0], orientation[1], orientation[2], orientation[3]
                        
                        siny_cosp = 2.0 * (qw * qy + qx * qz)
                        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
                        self.raw_yaw = float(np.degrees(np.arctan2(siny_cosp, cosy_cosp)))
                        self.yaw = float(
                            (self.raw_yaw * self.config.yaw_sign + self.config.heading_offset_deg + 180.0)
                            % 360.0
                            - 180.0
                        )
                        
                        # pitch
                        sinp = 2.0 * (qw * qx - qy * qz)
                        if abs(sinp) >= 1:
                            self.pitch = float(np.sign(sinp) * 90.0)
                        else:
                            self.pitch = float(np.degrees(np.arcsin(sinp)))
                            
                        # roll
                        sinr_cosp = 2.0 * (qw * qz + qx * qy)
                        cosr_cosp = 1.0 - 2.0 * (qx * qx + qz * qz)
                        self.roll = float(np.degrees(np.arctan2(sinr_cosp, cosr_cosp)))
                except Exception as pe:
                    print(f"[ZedDepthProcessor ERROR] Positional tracking retrieval failed: {pe}")
            
            # Calculate FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_time > 1.0:
                self.fps = self.frame_count / (current_time - self.last_fps_time)
                self.frame_count = 0
                self.last_fps_time = current_time
            return True
        else:
            self.none_frame_count += 1
            time_since_last = time.time() - self.last_frame_time
            if time_since_last > 5.0:
                print(f"[ZedDepthProcessor] WARNING: Camera connection frozen! No frames for {time_since_last:.1f}s")
                self.last_frame_time = time.time()  # reset threshold to prevent log spam
            return False
            
    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Retrieve the latest depth frame in millimeters"""
        if not self.zed:
            return None
            
        # retrieve depth measure
        self.zed.retrieve_measure(self.depth_map, sl.MEASURE.DEPTH)
        depth_data = self.depth_map.get_data()
        
        # Replace inf and nan values with 0 or max_depth
        if depth_data is not None:
            # Create a copy so we don't modify the memory map of sl.Mat directly
            depth_np = np.copy(depth_data)
            depth_np[np.isnan(depth_np)] = 0
            depth_np[np.isinf(depth_np)] = 0
            return depth_np
        return None
        
    def get_rgb_frame(self) -> Optional[np.ndarray]:
        """Retrieve the latest RGB frame in BGR format (OpenCV standard)"""
        if not self.zed:
            return None
            
        self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT)
        bgra_data = self.image_left.get_data()
        if bgra_data is not None:
            # bgra_data is RGBA or BGRA. ZED SDK get_data() for sl.VIEW.LEFT returns BGRA.
            # Convert to standard 3-channel BGR for OpenCV
            return cv2.cvtColor(bgra_data, cv2.COLOR_BGRA2BGR)
        return None

    def process_depth_for_navigation(self, depth_frame: np.ndarray) -> Dict:
        """
        Process depth frame for navigation decisions.
        Returns a dictionary containing grids, zones, obstacles, and safe directions.
        """
        if depth_frame is None:
            return None
        self.nav_frame_count += 1
            
        # Prepare navigation depth without turning invalid zeros into near obstacles.
        depth_clipped = self.prepare_navigation_depth(depth_frame)
        
        # Create grid-based depth map
        grid_depth = self.create_depth_grid(depth_clipped)
        
        # Analyze depth zones
        zones = self.analyze_depth_zones(depth_clipped)
        
        # Full obstacle maps are optional because guidance already uses zones and grid depth.
        obstacles = self.detect_obstacles(depth_clipped) if self.config.enable_obstacle_map else {}
        
        # Calculate safe directions
        safe_directions = self.calculate_safe_directions(grid_depth)
        
        # Accumulate 2D occupancy grid SLAM map
        if (
            self.positional_tracking_enabled
            and self.nav_frame_count % max(1, int(self.config.map_update_stride)) == 0
        ):
            self._accumulate_grid(grid_depth)
        
        return {
            'depth_frame': depth_frame,
            'depth_clipped': depth_clipped,
            'grid_depth': grid_depth,
            'zones': zones,
            'obstacles': obstacles,
            'safe_directions': safe_directions,
            'fps': self.fps,
            'timestamp': time.time()
        }

    def prepare_navigation_depth(self, depth_frame: np.ndarray) -> np.ndarray:
        """Sanitize raw ZED depth for navigation.

        ZED returns 0/NaN/Inf when stereo cannot estimate depth, often in long
        hallways or empty rooms. For navigation, those pixels should behave as
        far/open space instead of being clipped up to min_depth as a fake close
        obstacle.
        """
        depth_nav = np.array(depth_frame, dtype=np.float32, copy=True)
        valid = np.isfinite(depth_nav) & (depth_nav > 0)

        if self.config.invalid_depth_as_far:
            depth_nav[~valid] = float(self.config.max_depth)
        else:
            depth_nav[~valid] = 0.0

        valid = depth_nav > 0
        too_near = valid & (depth_nav < self.config.min_depth)
        too_far = valid & (depth_nav > self.config.max_depth)
        depth_nav[too_near] = float(self.config.min_depth)
        depth_nav[too_far] = float(self.config.max_depth)
        return depth_nav

    def _accumulate_grid(self, grid_depth: np.ndarray) -> None:
        """Projects depth points to gravity-aligned World coordinates, filters out floor/ceiling, and runs ray-clearing."""
        ty_m = self.ty / 1000.0
        map_yaw_deg = base_yaw_to_projection_yaw(self.yaw)
        
        # Grid coordinates of the user (center is 50,50)
        user_grid_z, user_grid_x = pose_mm_to_grid(
            self.tx,
            self.tz,
            cell_size_m=self.config.cell_size_m,
            width=100,
            height=100,
        )

        def project_camera_point_to_grid(x_c: float, z_c: float) -> tuple[int, int]:
            return camera_point_to_grid(
                user_grid_x,
                user_grid_z,
                map_yaw_deg,
                x_c,
                z_c,
                cell_size_m=self.config.cell_size_m,
                width=100,
                height=100,
            )
        
        # ZED 1 FOV parameters
        hfov = np.radians(90.0)
        vfov = np.radians(60.0)
        h, w = grid_depth.shape
        
        # Ray clearing range (m)
        d_clear_limit = float(self.config.ray_clear_limit_m)

        for col in range(w):
            col_angle = -hfov / 2.0 + col * (hfov / (w - 1))
            
            closest_obstacle_d = None
            closest_obstacle_pt = None
            
            for row in range(h):
                row_angle = vfov / 2.0 - row * (vfov / (h - 1))
                d_mm = grid_depth[row, col]
                if d_mm < self.config.min_depth or d_mm > self.config.max_depth:
                    continue
                if self.config.invalid_depth_as_far and d_mm >= self.config.max_depth * 0.98:
                    continue
                    
                d_m = d_mm / 1000.0
                
                # Keep camera lateral mirroring separate from VSLAM movement/facing.
                x_c = self.config.camera_x_sign * d_m * np.sin(col_angle)
                y_c = d_m * np.sin(row_angle)
                z_c = d_m * np.cos(col_angle) * np.cos(row_angle)
                
                y_w = ty_m + y_c
                
                # Obstacle height filter: ignore points near floor (e.g. y_world < -0.8m)
                # and points high up (above 0.4m relative to starting camera height)
                if -0.8 <= y_w <= 0.4:
                    if closest_obstacle_d is None or d_m < closest_obstacle_d:
                        closest_obstacle_d = d_m
                        grid_x, grid_z = project_camera_point_to_grid(x_c, z_c)
                        closest_obstacle_pt = (grid_x, y_w, grid_z)
            
            if closest_obstacle_pt is not None:
                grid_x, _, grid_z = closest_obstacle_pt
                
                # Raycast: clear all cells between user and obstacle
                steps = max(abs(grid_x - user_grid_x), abs(grid_z - user_grid_z))
                if steps > 0:
                    xs = np.linspace(user_grid_x, grid_x, steps + 1)[:-1]
                    zs = np.linspace(user_grid_z, grid_z, steps + 1)[:-1]
                    for px, pz in zip(xs, zs):
                        self._mark_free_cell(int(pz), int(px))
                
                # Mark obstacle cell
                self._mark_wall_cell(grid_z, grid_x)
            else:
                # No obstacle in this direction: clear path up to clearing limit.
                x_c = self.config.camera_x_sign * d_clear_limit * np.sin(col_angle)
                z_c = d_clear_limit * np.cos(col_angle)
                grid_x, grid_z = project_camera_point_to_grid(x_c, z_c)
                
                steps = max(abs(grid_x - user_grid_x), abs(grid_z - user_grid_z))
                if steps > 0:
                    xs = np.linspace(user_grid_x, grid_x, steps + 1)
                    zs = np.linspace(user_grid_z, grid_z, steps + 1)
                    for px, pz in zip(xs, zs):
                        self._mark_free_cell(int(pz), int(px))

        self._refresh_occupancy_from_confidence()
        
    def create_depth_grid(self, depth_frame: np.ndarray) -> np.ndarray:
        """Divide depth frame into grid and compute average depth per cell"""
        h, w = depth_frame.shape
        grid_h, grid_w = self.config.grid_height, self.config.grid_width
        
        cell_h = h // grid_h
        cell_w = w // grid_w
        if cell_h <= 0 or cell_w <= 0:
            return np.full((grid_h, grid_w), self.config.max_depth, dtype=np.float32)

        usable_h = cell_h * grid_h
        usable_w = cell_w * grid_w
        cells = depth_frame[:usable_h, :usable_w].reshape(grid_h, cell_h, grid_w, cell_w)

        if np.any(cells <= 0):
            masked = np.where(cells > 0, cells, np.nan)
            grid = np.nanmedian(masked, axis=(1, 3))
            grid = np.where(np.isnan(grid), self.config.max_depth, grid)
        else:
            grid = np.median(cells, axis=(1, 3))

        return grid.astype(np.float32, copy=False)
        
    def analyze_depth_zones(self, depth_frame: np.ndarray) -> Dict:
        """Analyze depth in different zones (left, center, right, near, far)"""
        h, w = depth_frame.shape
        
        # Divide into left, center, right
        left = depth_frame[:, :w//3]
        center = depth_frame[:, w//3:2*w//3]
        right = depth_frame[:, 2*w//3:]
        
        # Focus on lower half (ground level and knee height)
        lower_half_y = h // 2
        
        # Calculate stable depth profile vectorized per column.
        lower_depth = depth_frame[lower_half_y:, :]
        column_values = np.where(lower_depth > 0, lower_depth, np.nan)
        col_medians = np.nanmedian(column_values, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)

        # Smooth the profile using a 15-pixel moving average window.
        window = 15
        valid_cols = col_medians > 0
        kernel = np.ones(window, dtype=np.float32)
        smoothed_sum = np.convolve(np.where(valid_cols, col_medians, 0.0), kernel, mode="same")
        smoothed_count = np.convolve(valid_cols.astype(np.float32), kernel, mode="same")
        smoothed = np.divide(
            smoothed_sum,
            smoothed_count,
            out=np.zeros_like(smoothed_sum, dtype=np.float32),
            where=smoothed_count > 0,
        )
                
        if len(smoothed) > 0 and np.max(smoothed) > 0:
            global_max_val = float(np.max(smoothed))
            global_max_col = int(np.argmax(smoothed))
            if global_max_col < w // 3:
                global_max_zone = "left"
            elif global_max_col < 2 * w // 3:
                global_max_zone = "center"
            else:
                global_max_zone = "right"
                
            # Scan left to find the wall/obstacle boundary
            left_anomaly_val = 0.0
            for c in range(global_max_col - 1, -1, -1):
                if 0 < smoothed[c] < global_max_val * 0.6:
                    left_anomaly_val = float(smoothed[c])
                    break
            if left_anomaly_val == 0.0:
                left_valid = col_medians[:w//3]
                left_valid = left_valid[left_valid > 0]
                left_anomaly_val = float(np.min(left_valid)) if len(left_valid) > 0 else 465.0
                
            # Scan right to find the wall/obstacle boundary
            right_anomaly_val = 0.0
            for c in range(global_max_col + 1, w):
                if 0 < smoothed[c] < global_max_val * 0.6:
                    right_anomaly_val = float(smoothed[c])
                    break
            if right_anomaly_val == 0.0:
                right_valid = col_medians[2*w//3:]
                right_valid = right_valid[right_valid > 0]
                right_anomaly_val = float(np.min(right_valid)) if len(right_valid) > 0 else 781.0
        else:
            global_max_val = self.config.max_depth
            global_max_col = w // 2
            global_max_zone = "center"
            left_anomaly_val = 465.0
            right_anomaly_val = 781.0
            
        def get_zone_stats(zone):
            valid = zone[zone > 0]
            if len(valid) == 0:
                return {
                    'min': self.config.max_depth,
                    'mean': self.config.max_depth,
                    'median': self.config.max_depth,
                    'open_space_ratio': 1.0,
                }
            open_space_ratio = float(np.mean(valid >= self.config.max_depth * 0.95))
            return {
                'min': float(np.min(valid)),
                'mean': float(np.mean(valid)),
                'median': float(np.median(valid)),
                'open_space_ratio': open_space_ratio,
            }
            
        return {
            'left': get_zone_stats(left[lower_half_y:, :]),
            'center': get_zone_stats(center[lower_half_y:, :]),
            'right': get_zone_stats(right[lower_half_y:, :]),
            'full': get_zone_stats(depth_frame[lower_half_y:, :]),
            'global_depth_max': {
                'value': global_max_val,
                'col': global_max_col,
                'zone': global_max_zone,
                'left_wall_anomaly_mm': left_anomaly_val,
                'right_wall_anomaly_mm': right_anomaly_val
            }
        }
        
    def detect_obstacles(self, depth_frame: np.ndarray, 
                        obstacle_threshold: int = 1000) -> Dict:
        """Detect obstacles closer than threshold (default 1m)"""
        # Create binary obstacle map
        obstacle_map = (depth_frame > 0) & (depth_frame < obstacle_threshold)
        
        # Count obstacles in different regions
        h, w = depth_frame.shape
        left_obstacles = np.sum(obstacle_map[:, :w//3])
        center_obstacles = np.sum(obstacle_map[:, w//3:2*w//3])
        right_obstacles = np.sum(obstacle_map[:, 2*w//3:])
        
        total_pixels = h * w
        obstacle_percentage = (np.sum(obstacle_map) / total_pixels) * 100
        
        return {
            'obstacle_map': obstacle_map,
            'left_count': int(left_obstacles),
            'center_count': int(center_obstacles),
            'right_count': int(right_obstacles),
            'total_percentage': float(obstacle_percentage),
            'has_obstacle': obstacle_percentage > 5.0  # 5% threshold
        }
        
    def calculate_safe_directions(self, grid_depth: np.ndarray, 
                                  safe_distance: int = 1500) -> Dict:
        """Calculate which directions are safe to move"""
        grid_h, grid_w = grid_depth.shape
        
        # Focus on bottom third (ground level)
        ground_level = grid_depth[2*grid_h//3:, :]
        
        # Calculate average depth for each column
        column_depths = np.mean(ground_level, axis=0)
        
        # Divide into 5 sectors: far-left, left, center, right, far-right
        sector_size = grid_w // 5
        sectors = []
        for i in range(5):
            start = i * sector_size
            end = (i + 1) * sector_size if i < 4 else grid_w
            sector_depth = np.mean(column_depths[start:end])
            is_safe = sector_depth > safe_distance
            sectors.append({
                'depth': float(sector_depth),
                'safe': bool(is_safe),
                'score': float(min(sector_depth / safe_distance, 2.0))
            })
            
        return {
            'far_left': sectors[0],
            'left': sectors[1],
            'center': sectors[2],
            'right': sectors[3],
            'far_right': sectors[4],
            'best_direction': self._get_best_direction(sectors)
        }
        
    def _get_best_direction(self, sectors: list) -> str:
        """Determine best direction from sector analysis"""
        direction_names = ['far_left', 'left', 'center', 'right', 'far_right']
        preference_order = [2, 1, 3, 0, 4]  # center, left, right, far_left, far_right
        
        for idx in preference_order:
            if sectors[idx]['safe']:
                return direction_names[idx]
                
        # If nothing is safe, return direction with maximum depth
        max_idx = max(range(5), key=lambda i: sectors[i]['depth'])
        return direction_names[max_idx]
        
    def visualize_depth(self, depth_frame: np.ndarray) -> np.ndarray:
        """Create colorized visualization of depth frame"""
        if depth_frame is None:
            return None
            
        # Normalize to 0-255 range (ignoring zeros for normalization)
        depth_copy = np.copy(depth_frame)
        depth_copy[depth_copy == 0] = self.config.max_depth
        
        depth_normalized = np.interp(depth_copy, 
                                    (self.config.min_depth, self.config.max_depth), 
                                    (0, 255)).astype(np.uint8)
                                    
        # Apply colormap (Turbo has excellent contrast: red=close, blue=far)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)
        
        # Mask out invalid depth pixels (set them to black)
        depth_colored[depth_frame == 0] = [0, 0, 0]
        
        return depth_colored
        
    def __enter__(self):
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


if __name__ == "__main__":
    """Test ZED Depth Processor (simple command-line test)"""
    print("Initializing ZED camera test...")
    processor = ZedDepthProcessor()
    try:
        processor.start()
        print("Camera started successfully. Grab 10 test frames...")
        for i in range(10):
            if processor.grab_frame():
                depth = processor.get_depth_frame()
                rgb = processor.get_rgb_frame()
                if depth is not None:
                    print(f"Frame {i+1}: Grabbed! Shape={depth.shape}, Median Depth={np.median(depth):.1f}mm")
            time.sleep(0.1)
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        processor.stop()
