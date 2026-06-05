import copy
import heapq
import json
import math
import time
import uuid
from pathlib import Path


STATIC_LANDMARK_CLASSES = {
    "door",
    "doorway",
    "exit sign",
    "stairs",
    "shelf",
    "counter",
    "table",
    "dining table",
    "bench",
}

STATIC_OBJECT_CLASSES = {
    "door",
    "doorway",
    "door local",
    "door_local",
    "open door",
    "open_door",
    "closed door",
    "closed_door",
    "exit sign",
    "exit_sign",
    "stairs",
    "shelf",
    "counter",
    "table",
    "dining table",
    "bench",
    "chair",
    "couch",
    "bed",
    "tv",
    "potted plant",
}

DYNAMIC_CLASSES = {
    "person",
    "dog",
    "cat",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "box",
    "cell phone",
    "laptop",
    "book",
    "keyboard",
    "mouse",
    "remote",
}

EXIT_LANDMARK_TYPES = {"door", "doorway", "exit sign"}
OBJECT_CLASSIFICATION_COLORS = {
    "static": "#38bdf8",
    "dynamic": "#f97316",
    "landmark": "#facc15",
    "exit": "#22f59c",
}
EXIT_REACHED_RADIUS_CELLS = 4
MOVING_AWAY_DELTA_CELLS = 3


def _clamp(value, low, high):
    return max(low, min(value, high))


def _normalize_label(label):
    return str(label or "").strip().lower().replace("_", " ").replace("-", " ")


def classify_object_mobility(label):
    normalized = _normalize_label(label)
    if normalized in STATIC_OBJECT_CLASSES:
        return "static"
    if normalized in DYNAMIC_CLASSES:
        return "dynamic"
    return "dynamic"


def object_color_for_label(label, mobility=None):
    normalized = _normalize_label(label)
    if "exit" in normalized:
        return OBJECT_CLASSIFICATION_COLORS["exit"]
    if "door" in normalized:
        return OBJECT_CLASSIFICATION_COLORS["landmark"]
    resolved_mobility = mobility or classify_object_mobility(normalized)
    return OBJECT_CLASSIFICATION_COLORS.get(resolved_mobility, OBJECT_CLASSIFICATION_COLORS["dynamic"])


class SpatialMemoryNavigationSystem:
    def __init__(self, maps_dir=None, cell_size_m=0.10, width=100, height=100):
        self.maps_dir = Path(maps_dir or Path(__file__).resolve().parent / "maps")
        self.maps_dir.mkdir(parents=True, exist_ok=True)
        self.cell_size_m = cell_size_m
        self.width = width
        self.height = height
        self.map_graph_path = self.maps_dir / "map_graph.json"
        self.map_graph = self._load_map_graph()
        self.state = {
            "mode": "idle",
            "current_map_id": None,
            "current_map": None,
            "user_pose": {"x": 0.0, "z": 0.0, "yaw": 0.0},
            "active_goal": None,
            "active_path": [],
            "current_waypoint_index": 0,
            "live_dynamic_obstacles": [],
            "last_instruction": "",
            "previous_user_pos": None,
            "navigation_target": None,
            "last_goal_distance_cells": None,
            "exit_reached": False,
            "map_session_origin_grid": None,
            "map_session_origin_pose": None,
        }

    def _load_map_graph(self):
        if not self.map_graph_path.exists():
            return {}
        try:
            return json.loads(self.map_graph_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_map_graph(self):
        self.map_graph_path.write_text(json.dumps(self.map_graph, indent=2), encoding="utf-8")

    def _map_path(self, map_id):
        safe_id = str(map_id).replace("/", "_").replace("\\", "_")
        return self.maps_dir / f"{safe_id}.json"

    def _normalize_grid(self, grid):
        if not grid:
            return [[2 for _ in range(self.width)] for _ in range(self.height)]

        normalized = []
        for r in range(self.height):
            source_row = grid[r] if r < len(grid) and isinstance(grid[r], list) else []
            row = []
            for c in range(self.width):
                value = source_row[c] if c < len(source_row) else 2
                try:
                    value = int(value)
                except Exception:
                    value = 2
                row.append(_clamp(value, 0, 2))
            normalized.append(row)
        return normalized

    def _raw_pose_to_grid(self, pose):
        x_m = float(pose.get("x", 0.0)) / 1000.0
        z_m = float(pose.get("z", 0.0)) / 1000.0
        gx = _clamp(int(x_m / self.cell_size_m) + self.width // 2, 0, self.width - 1)
        gz = _clamp(int(z_m / self.cell_size_m) + self.height // 2, 0, self.height - 1)
        return gz, gx

    def _origin_grid_for_map(self, map_data):
        session_origin = self.state.get("map_session_origin_grid")
        current_map = self.state.get("current_map")
        if (
            session_origin
            and current_map
            and map_data
            and current_map.get("map_id") == map_data.get("map_id")
        ):
            return (
                _clamp(int(session_origin.get("z", self.height // 2)), 0, self.height - 1),
                _clamp(int(session_origin.get("x", self.width // 2)), 0, self.width - 1),
            )
        metadata = (map_data or {}).get("metadata", {})
        origin_grid = metadata.get("origin_grid") or {}
        return (
            _clamp(int(origin_grid.get("z", self.height // 2)), 0, self.height - 1),
            _clamp(int(origin_grid.get("x", self.width // 2)), 0, self.width - 1),
        )

    def _to_map_grid_cell(self, raw_cell, map_data):
        origin_z, origin_x = self._origin_grid_for_map(map_data)
        raw_z, raw_x = raw_cell
        map_z = _clamp(raw_z - origin_z + self.height // 2, 0, self.height - 1)
        map_x = _clamp(raw_x - origin_x + self.width // 2, 0, self.width - 1)
        return map_z, map_x

    def pose_to_grid(self, pose):
        raw_cell = self._raw_pose_to_grid(pose)
        current_map = self.state.get("current_map")
        if current_map:
            return self._to_map_grid_cell(raw_cell, current_map)
        return raw_cell

    def _grid_to_map_frame(self, grid, map_data):
        normalized = self._normalize_grid(grid)
        origin_z, origin_x = self._origin_grid_for_map(map_data)
        shifted = [[0 for _ in range(self.width)] for _ in range(self.height)]
        for raw_z in range(self.height):
            for raw_x in range(self.width):
                value = int(normalized[raw_z][raw_x])
                if value == 0:
                    continue
                map_z = raw_z - origin_z + self.height // 2
                map_x = raw_x - origin_x + self.width // 2
                if 0 <= map_z < self.height and 0 <= map_x < self.width:
                    shifted[map_z][map_x] = value
        return shifted

    def _semantic_objects_to_map_frame(self, semantic_objects, map_data):
        mapped = []
        for obj in semantic_objects or []:
            item = copy.deepcopy(obj)
            gx = item.get("x")
            gz = item.get("z")
            if gx is not None and gz is not None:
                map_z, map_x = self._to_map_grid_cell((int(gz), int(gx)), map_data)
                item["x"] = map_x
                item["z"] = map_z
                item["raw_x"] = int(gx)
                item["raw_z"] = int(gz)
            mapped.append(item)
        return mapped

    def semantic_objects_to_current_map(self, semantic_objects):
        current_map = self.state.get("current_map")
        if not current_map:
            return [copy.deepcopy(obj) for obj in semantic_objects or []]
        return self._semantic_objects_to_map_frame(semantic_objects, current_map)

    def _update_user_pose(self, pose):
        self.state["user_pose"] = {
            "x": float(pose.get("x", 0.0)) / 1000.0,
            "z": float(pose.get("z", 0.0)) / 1000.0,
            "yaw": float(pose.get("yaw", 0.0)),
        }

    def _create_empty_map(self, map_name, static_grid=None, map_id=None, origin_pose=None, origin_grid=None):
        map_id = map_id or f"{map_name.lower().replace(' ', '_')}_{int(time.time())}"
        return {
            "map_id": map_id,
            "map_name": map_name,
            "created_at": time.time(),
            "cell_size_m": self.cell_size_m,
            "width": self.width,
            "height": self.height,
            "static_grid": self._normalize_grid(static_grid),
            "landmarks": [],
            "static_objects": [],
            "metadata": {
                "scan_quality": 0.0,
                "coverage_percent": 0.0,
                "origin_pose_mm": origin_pose or {"x": 0.0, "z": 0.0, "yaw": 0.0},
                "origin_grid": origin_grid or {"z": self.height // 2, "x": self.width // 2},
            },
        }

    def _grid_coverage(self, grid):
        total = self.width * self.height
        known = sum(1 for row in grid for cell in row if int(cell) != 2)
        return round((known / total) * 100.0, 2)

    def _grid_counts(self, grid):
        counts = {"free": 0, "occupied": 0, "special": 0, "unknown": 0}
        for row in self._normalize_grid(grid):
            for cell in row:
                value = int(cell)
                if value == 0:
                    counts["free"] += 1
                elif value == 1:
                    counts["occupied"] += 1
                elif value == 2:
                    counts["special"] += 1
                else:
                    counts["unknown"] += 1
        return counts

    def _merge_static_grid(self, existing_grid, live_grid):
        existing = self._normalize_grid(existing_grid)
        live = self._normalize_grid(live_grid)
        merged = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                live_value = int(live[r][c])
                existing_value = int(existing[r][c])
                row.append(live_value if live_value != 0 else existing_value)
            merged.append(row)
        return merged

    def _add_or_update_landmark(self, map_data, landmark):
        for existing in map_data["landmarks"]:
            if existing["type"] != landmark["type"]:
                continue
            dist = math.hypot(
                existing["grid_x"] - landmark["grid_x"],
                existing["grid_z"] - landmark["grid_z"],
            )
            if dist <= 3:
                existing["grid_x"] = int(round((existing["grid_x"] + landmark["grid_x"]) / 2))
                existing["grid_z"] = int(round((existing["grid_z"] + landmark["grid_z"]) / 2))
                existing["confidence"] = max(existing.get("confidence", 0.0), landmark["confidence"])
                return existing

        map_data["landmarks"].append(landmark)
        return landmark

    def _add_or_update_static_object(self, map_data, static_object):
        map_data.setdefault("static_objects", [])
        for existing in map_data["static_objects"]:
            if existing["label"] != static_object["label"]:
                continue
            dist = math.hypot(
                existing["grid_x"] - static_object["grid_x"],
                existing["grid_z"] - static_object["grid_z"],
            )
            if dist <= 3:
                existing["grid_x"] = int(round((existing["grid_x"] + static_object["grid_x"]) / 2))
                existing["grid_z"] = int(round((existing["grid_z"] + static_object["grid_z"]) / 2))
                existing["confidence"] = max(existing.get("confidence", 0.0), static_object["confidence"])
                if static_object.get("distance") is not None:
                    existing["distance"] = static_object["distance"]
                return existing

        map_data["static_objects"].append(static_object)
        return static_object

    def update_static_objects(self, map_data, semantic_objects):
        map_data.setdefault("static_objects", [])
        for obj in semantic_objects or []:
            label = _normalize_label(obj.get("label") or obj.get("class") or obj.get("class_name"))
            mobility = obj.get("mobility") or obj.get("classification") or classify_object_mobility(label)
            if mobility != "static" or label not in STATIC_OBJECT_CLASSES:
                continue
            gx = obj.get("x")
            gz = obj.get("z")
            if gx is None or gz is None:
                continue
            static_object = {
                "id": f"{label.replace(' ', '_')}_{uuid.uuid4().hex[:8]}",
                "label": "table" if label == "dining table" else label,
                "detected_label": _normalize_label(obj.get("detected_label") or label),
                "classification": "static",
                "mobility": "static",
                "grid_x": _clamp(int(gx), 0, self.width - 1),
                "grid_z": _clamp(int(gz), 0, self.height - 1),
                "confidence": float(obj.get("confidence", 0.75)),
                "distance": obj.get("distance"),
                "source_model": obj.get("source_model"),
                "color": object_color_for_label(label, "static"),
            }
            self._add_or_update_static_object(map_data, static_object)

    def update_static_landmarks(self, map_data, semantic_objects):
        for obj in semantic_objects or []:
            label = _normalize_label(obj.get("label") or obj.get("class") or obj.get("class_name"))
            if label not in STATIC_LANDMARK_CLASSES:
                continue
            gx = obj.get("x")
            gz = obj.get("z")
            if gx is None or gz is None:
                continue
            landmark_type = "table" if label == "dining table" else label
            landmark = {
                "id": f"{landmark_type.replace(' ', '_')}_{uuid.uuid4().hex[:8]}",
                "type": landmark_type,
                "grid_x": _clamp(int(gx), 0, self.width - 1),
                "grid_z": _clamp(int(gz), 0, self.height - 1),
                "confidence": float(obj.get("confidence", 0.75)),
                "target_map_id": obj.get("target_map_id"),
                "target_door_id": obj.get("target_door_id"),
            }
            self._add_or_update_landmark(map_data, landmark)

    def get_live_dynamic_obstacles(self, semantic_objects):
        obstacles = []
        for obj in semantic_objects or []:
            label = _normalize_label(obj.get("label") or obj.get("class") or obj.get("class_name"))
            mobility = obj.get("mobility") or obj.get("classification") or classify_object_mobility(label)
            if mobility != "dynamic" or label not in DYNAMIC_CLASSES:
                continue
            gx = obj.get("x")
            gz = obj.get("z")
            if gx is None or gz is None:
                continue
            obstacles.append({
                "class_name": label,
                "classification": "dynamic",
                "mobility": "dynamic",
                "color": object_color_for_label(label, "dynamic"),
                "grid_x": _clamp(int(gx), 0, self.width - 1),
                "grid_z": _clamp(int(gz), 0, self.height - 1),
                "distance": obj.get("distance"),
            })
        self.state["live_dynamic_obstacles"] = obstacles
        return obstacles

    def start_mapping(self, map_name, live_grid, semantic_objects, pose, map_id=None):
        raw_origin_z, raw_origin_x = self._raw_pose_to_grid(pose)
        origin_grid = {"z": raw_origin_z, "x": raw_origin_x}
        origin_pose = {
            "x": float(pose.get("x", 0.0)),
            "z": float(pose.get("z", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
        }
        map_context = {"metadata": {"origin_grid": origin_grid}}
        map_grid = self._grid_to_map_frame(live_grid, map_context)
        current_map = self._create_empty_map(
            map_name,
            map_grid,
            map_id=map_id,
            origin_pose=origin_pose,
            origin_grid=origin_grid,
        )
        semantic_objects_map = self._semantic_objects_to_map_frame(semantic_objects, current_map)
        self.update_static_objects(current_map, semantic_objects_map)
        self.update_static_landmarks(current_map, semantic_objects_map)
        current_map["metadata"]["grid_counts"] = self._grid_counts(current_map["static_grid"])
        current_map["metadata"]["coverage_percent"] = self._grid_coverage(current_map["static_grid"])
        current_map["metadata"]["scan_quality"] = min(1.0, current_map["metadata"]["coverage_percent"] / 85.0)
        current_map["metadata"]["static_object_count"] = len(current_map.get("static_objects", []))
        self.state.update({
            "mode": "mapping",
            "current_map_id": current_map["map_id"],
            "current_map": current_map,
            "active_goal": None,
            "active_path": [],
            "current_waypoint_index": 0,
            "map_session_origin_grid": origin_grid,
            "map_session_origin_pose": origin_pose,
        })
        self._update_user_pose(pose)
        return current_map

    def refresh_mapping(self, live_grid, semantic_objects):
        current_map = self.state.get("current_map")
        if not current_map:
            return None
        live_grid_map = self._grid_to_map_frame(live_grid, current_map)
        semantic_objects_map = self._semantic_objects_to_map_frame(semantic_objects, current_map)
        current_map["static_grid"] = self._merge_static_grid(current_map.get("static_grid"), live_grid_map)
        self.update_static_objects(current_map, semantic_objects_map)
        self.update_static_landmarks(current_map, semantic_objects_map)
        current_map["metadata"]["grid_counts"] = self._grid_counts(current_map["static_grid"])
        current_map["metadata"]["coverage_percent"] = self._grid_coverage(current_map["static_grid"])
        current_map["metadata"]["scan_quality"] = min(1.0, current_map["metadata"]["coverage_percent"] / 85.0)
        current_map["metadata"]["static_object_count"] = len(current_map.get("static_objects", []))
        return current_map

    def save_current_map(self, live_grid=None, semantic_objects=None):
        current_map = self.state.get("current_map")
        if not current_map:
            return None
        if live_grid is not None:
            self.refresh_mapping(live_grid, semantic_objects or [])
        path = self._map_path(current_map["map_id"])
        path.write_text(json.dumps(current_map, indent=2), encoding="utf-8")
        self.map_graph.setdefault(current_map["map_id"], {
            "name": current_map["map_name"],
            "exits": [],
        })
        self._save_map_graph()
        return current_map

    def stop_mapping(self):
        if self.state["mode"] == "mapping":
            self.state["mode"] = "idle"
        return self.state

    def list_maps(self):
        maps = []
        for path in sorted(self.maps_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if path.name == self.map_graph_path.name:
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            maps.append({
                "map_id": data.get("map_id"),
                "map_name": data.get("map_name"),
                "created_at": data.get("created_at"),
                "saved_at": path.stat().st_mtime,
                "file_name": path.name,
                "landmark_count": len(data.get("landmarks", [])),
                "static_object_count": len(data.get("static_objects", [])),
                "coverage_percent": data.get("metadata", {}).get("coverage_percent", 0.0),
            })
        return maps

    def load_map(self, map_id=None, map_name=None, anchor_pose=None):
        candidates = []
        if map_id:
            candidates.append(self._map_path(map_id))
        else:
            candidates = sorted(self.maps_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

        for path in candidates:
            if not path.exists() or path.name == self.map_graph_path.name:
                continue
            data = json.loads(path.read_text(encoding="utf-8"))
            if map_name and data.get("map_name", "").lower() != str(map_name).lower():
                continue
            data["static_grid"] = self._normalize_grid(data.get("static_grid"))
            data.setdefault("static_objects", [])
            data.setdefault("metadata", {})
            data["metadata"].setdefault("origin_pose_mm", {"x": 0.0, "z": 0.0, "yaw": 0.0})
            data["metadata"].setdefault("origin_grid", {"z": self.height // 2, "x": self.width // 2})
            data["metadata"]["static_object_count"] = len(data.get("static_objects", []))
            if anchor_pose is not None:
                raw_origin_z, raw_origin_x = self._raw_pose_to_grid(anchor_pose)
                session_origin_grid = {"z": raw_origin_z, "x": raw_origin_x}
                session_origin_pose = {
                    "x": float(anchor_pose.get("x", 0.0)),
                    "z": float(anchor_pose.get("z", 0.0)),
                    "yaw": float(anchor_pose.get("yaw", 0.0)),
                }
            else:
                session_origin_grid = data["metadata"].get("origin_grid")
                session_origin_pose = data["metadata"].get("origin_pose_mm")
            self.state["current_map_id"] = data.get("map_id")
            self.state["current_map"] = data
            self.state["map_session_origin_grid"] = session_origin_grid
            self.state["map_session_origin_pose"] = session_origin_pose
            return data
        return None

    def unload_map(self):
        self.stop_navigation()
        if self.state["mode"] == "mapping":
            self.state["mode"] = "idle"
        self.state["current_map_id"] = None
        self.state["current_map"] = None
        self.state["active_goal"] = None
        self.state["active_path"] = []
        self.state["current_waypoint_index"] = 0
        self.state["navigation_target"] = None
        self.state["map_session_origin_grid"] = None
        self.state["map_session_origin_pose"] = None
        return self.state

    def build_navigation_grid(self, static_grid, live_obstacles):
        nav_grid = copy.deepcopy(self._normalize_grid(static_grid))
        for obstacle in live_obstacles:
            gx = int(obstacle["grid_x"])
            gz = int(obstacle["grid_z"])
            nav_grid[gz][gx] = 1
            for dz in range(-3, 4):
                for dx in range(-3, 4):
                    if math.hypot(dx, dz) > 3:
                        continue
                    nz = gz + dz
                    nx = gx + dx
                    if 0 <= nz < self.height and 0 <= nx < self.width and nav_grid[nz][nx] != 1:
                        nav_grid[nz][nx] = max(float(nav_grid[nz][nx]), 0.7)
        return nav_grid

    def astar(self, nav_grid, start, goal):
        if not self._in_bounds(start) or not self._in_bounds(goal):
            return []
        if self._is_blocked(nav_grid, goal):
            return []

        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            for dz, dx in neighbors:
                neighbor = (current[0] + dz, current[1] + dx)
                if not self._in_bounds(neighbor) or self._is_blocked(nav_grid, neighbor):
                    continue
                base_cost = 1.414 if dz and dx else 1.0
                cell = float(nav_grid[neighbor[0]][neighbor[1]])
                risk_cost = 2.5 if cell == 2.0 else max(0.0, cell) * 3.0
                tentative = g_score[current] + base_cost + risk_cost
                if neighbor not in g_score or tentative < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score = tentative + math.hypot(neighbor[0] - goal[0], neighbor[1] - goal[1])
                    heapq.heappush(open_set, (f_score, neighbor))
        return []

    def _in_bounds(self, cell):
        z, x = cell
        return 0 <= z < self.height and 0 <= x < self.width

    def _is_blocked(self, nav_grid, cell):
        return int(float(nav_grid[cell[0]][cell[1]])) == 1

    def calculate_path_score(self, path, nav_grid, live_obstacles):
        if not path:
            return float("inf")
        path_length = max(0, len(path) - 1) * self.cell_size_m
        obstacle_risk = 0.0
        unknown_area_penalty = 0.0
        narrow_path_penalty = 0.0
        dynamic_obstacle_penalty = 0.0
        dynamic_cells = [(o["grid_z"], o["grid_x"]) for o in live_obstacles]

        for z, x in path:
            cell = float(nav_grid[z][x])
            if cell == 2.0:
                unknown_area_penalty += 1.0
            elif cell > 0:
                obstacle_risk += cell

            blocked_neighbors = 0
            for dz, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nz, nx = z + dz, x + dx
                if 0 <= nz < self.height and 0 <= nx < self.width and self._is_blocked(nav_grid, (nz, nx)):
                    blocked_neighbors += 1
            if blocked_neighbors >= 2:
                narrow_path_penalty += 1.0

            for oz, ox in dynamic_cells:
                dist = math.hypot(z - oz, x - ox)
                if dist <= 4:
                    dynamic_obstacle_penalty += (4 - dist) / 4.0

        return (
            path_length * 1.0
            + obstacle_risk * 3.0
            + narrow_path_penalty * 2.0
            + unknown_area_penalty * 2.5
            + dynamic_obstacle_penalty * 4.0
        )

    def choose_best_exit(self, user_pos, map_data, live_obstacles):
        doors = [
            lm for lm in map_data.get("landmarks", [])
            if _normalize_label(lm.get("type")) in EXIT_LANDMARK_TYPES
        ]
        if not doors:
            return None, "No saved exit found in this room."

        nav_grid = self.build_navigation_grid(map_data.get("static_grid"), live_obstacles)
        candidates = []
        for door in doors:
            goal = (int(door["grid_z"]), int(door["grid_x"]))
            path = self.astar(nav_grid, user_pos, goal)
            if path:
                candidates.append({
                    "door": door,
                    "path": path,
                    "score": self.calculate_path_score(path, nav_grid, live_obstacles),
                })
        if not candidates:
            return None, "All exits are currently blocked."
        return min(candidates, key=lambda item: item["score"]), None

    def start_navigation(self, goal_type, pose, semantic_objects, target_landmark_id=None):
        current_map = self.state.get("current_map")
        if not current_map:
            current_map = self.load_map(anchor_pose=pose)
        if not current_map:
            return None, "No saved map loaded."

        self._update_user_pose(pose)
        user_pos = self.pose_to_grid(pose)
        semantic_objects_map = self._semantic_objects_to_map_frame(semantic_objects, current_map)
        live_obstacles = self.get_live_dynamic_obstacles(semantic_objects_map)
        if goal_type == "exit":
            best, error = self.choose_best_exit(user_pos, current_map, live_obstacles)
        else:
            best, error = self._choose_landmark(user_pos, current_map, live_obstacles, target_landmark_id)
        if error:
            return None, error

        self.state.update({
            "mode": "navigating",
            "active_goal": (int(best["door"]["grid_z"]), int(best["door"]["grid_x"])),
            "active_path": best["path"],
            "current_waypoint_index": 0,
            "navigation_target": best["door"],
            "previous_user_pos": user_pos,
            "last_goal_distance_cells": math.hypot(
                user_pos[0] - int(best["door"]["grid_z"]),
                user_pos[1] - int(best["door"]["grid_x"]),
            ),
            "exit_reached": False,
            "last_instruction": "",
        })
        return best, None

    def _choose_landmark(self, user_pos, map_data, live_obstacles, target_landmark_id):
        landmarks = map_data.get("landmarks", [])
        if target_landmark_id:
            landmarks = [lm for lm in landmarks if lm.get("id") == target_landmark_id]
        if not landmarks:
            return None, "Target landmark not found."
        nav_grid = self.build_navigation_grid(map_data.get("static_grid"), live_obstacles)
        candidates = []
        for landmark in landmarks:
            goal = (int(landmark["grid_z"]), int(landmark["grid_x"]))
            path = self.astar(nav_grid, user_pos, goal)
            if path:
                candidates.append({
                    "door": landmark,
                    "path": path,
                    "score": self.calculate_path_score(path, nav_grid, live_obstacles),
                })
        if not candidates:
            return None, "Target is currently unreachable."
        return min(candidates, key=lambda item: item["score"]), None

    def navigation_guidance(self, pose, semantic_objects):
        if self.state["mode"] != "navigating" or not self.state.get("active_path"):
            return {
                "active": False,
                "instruction": "",
                "target": None,
                "remaining_distance_m": 0.0,
                "rerouted": False,
            }

        self._update_user_pose(pose)
        user_pos = self.pose_to_grid(pose)
        semantic_objects_map = self._semantic_objects_to_map_frame(semantic_objects, self.state["current_map"])
        live_obstacles = self.get_live_dynamic_obstacles(semantic_objects_map)
        nav_grid = self.build_navigation_grid(self.state["current_map"]["static_grid"], live_obstacles)
        rerouted = False
        warning = None
        goal_distance = self._goal_distance_cells(user_pos)

        if goal_distance is not None and goal_distance <= EXIT_REACHED_RADIUS_CELLS:
            target = self.state.get("navigation_target") or {}
            target_type = _normalize_label(target.get("type"))
            if target_type in EXIT_LANDMARK_TYPES:
                instruction = "You have reached the exit doorway. You have exited the room."
                self.state["exit_reached"] = True
                self.state["last_goal_distance_cells"] = goal_distance
                self.state["previous_user_pos"] = user_pos
                self.state["last_instruction"] = instruction
                return self._guidance_response(instruction, arrived=True)

        if self.is_moving_away_from_goal(user_pos):
            warning = self.return_to_goal_instruction(user_pos, float(pose.get("yaw", 0.0)))
            rerouted = True
        elif self.detect_wrong_direction(user_pos):
            warning = "You are moving away from the planned path. Replanning."
            rerouted = True
        elif self.path_blocked_by_live_obstacle(self.state["active_path"], live_obstacles):
            warning = "Obstacle detected. Rerouting."
            rerouted = True

        if rerouted:
            new_path = self.astar(nav_grid, user_pos, self.state["active_goal"])
            if new_path:
                self.state["active_path"] = new_path
                self.state["current_waypoint_index"] = 0
            else:
                instruction = "The route is blocked. Stop and scan around."
                self.state["last_instruction"] = instruction
                return self._guidance_response(instruction, rerouted=True)

        instruction = self.get_next_instruction(user_pos, float(pose.get("yaw", 0.0)))
        if warning:
            instruction = f"{warning} {instruction}"
        self.state["previous_user_pos"] = user_pos
        if goal_distance is not None:
            self.state["last_goal_distance_cells"] = goal_distance
        self.state["last_instruction"] = instruction
        return self._guidance_response(instruction, rerouted=rerouted)

    def _guidance_response(self, instruction, rerouted=False, arrived=False):
        path = self.state.get("active_path", [])
        idx = self.state.get("current_waypoint_index", 0)
        remaining_cells = max(0, len(path) - idx - 1)
        target = self.state.get("navigation_target") or {}
        user_pose = self.state.get("user_pose") or {}
        user_cell = self.pose_to_grid({
            "x": float(user_pose.get("x", 0.0)) * 1000.0,
            "z": float(user_pose.get("z", 0.0)) * 1000.0,
            "yaw": user_pose.get("yaw", 0.0),
        })
        distance_to_goal = self._goal_distance_cells(user_cell)
        return {
            "active": True,
            "instruction": instruction,
            "target": target.get("id"),
            "target_type": target.get("type"),
            "remaining_distance_m": round(remaining_cells * self.cell_size_m, 2),
            "distance_to_goal_m": round(distance_to_goal * self.cell_size_m, 2) if distance_to_goal is not None else None,
            "rerouted": rerouted,
            "arrived": arrived,
            "user_grid": {"z": user_cell[0], "x": user_cell[1]},
            "path": path,
            "goal": {"z": self.state["active_goal"][0], "x": self.state["active_goal"][1]},
        }

    def _goal_distance_cells(self, user_pos):
        goal = self.state.get("active_goal")
        if not goal:
            return None
        return math.hypot(user_pos[0] - goal[0], user_pos[1] - goal[1])

    def is_moving_away_from_goal(self, user_pos):
        current_distance = self._goal_distance_cells(user_pos)
        previous_distance = self.state.get("last_goal_distance_cells")
        if current_distance is None or previous_distance is None:
            return False
        if current_distance <= EXIT_REACHED_RADIUS_CELLS:
            return False
        return current_distance > previous_distance + MOVING_AWAY_DELTA_CELLS

    def return_to_goal_instruction(self, user_pos, user_yaw):
        target = self.state.get("navigation_target") or {}
        target_name = _normalize_label(target.get("type")) or "target"
        goal = self.state.get("active_goal")
        if not goal:
            return f"Moving away from the {target_name}. Replanning."

        dz = goal[0] - user_pos[0]
        dx = goal[1] - user_pos[1]
        target_angle = math.degrees(math.atan2(dx, dz))
        delta = (target_angle - user_yaw + 540.0) % 360.0 - 180.0

        if abs(delta) <= 25:
            turn_instruction = "Go straight to return to it"
        elif 25 < delta <= 135:
            turn_instruction = "Turn right to return to it"
        elif -135 <= delta < -25:
            turn_instruction = "Turn left to return to it"
        else:
            turn_instruction = "Turn around to return to it"
        return f"Moving away from the {target_name}. {turn_instruction}."

    def detect_wrong_direction(self, user_pos):
        previous = self.state.get("previous_user_pos")
        path = self.state.get("active_path") or []
        if not previous or not path:
            return False
        old_distance = self.distance_to_path(previous, path)
        new_distance = self.distance_to_path(user_pos, path)
        return new_distance > old_distance + 5

    def distance_to_path(self, pos, path):
        return min(math.hypot(pos[0] - z, pos[1] - x) for z, x in path)

    def path_blocked_by_live_obstacle(self, path, live_obstacles):
        for obstacle in live_obstacles:
            oz = obstacle["grid_z"]
            ox = obstacle["grid_x"]
            for z, x in path:
                if math.hypot(z - oz, x - ox) <= 2:
                    return True
        return False

    def get_next_instruction(self, user_pos, user_yaw):
        path = self.state.get("active_path", [])
        if not path:
            return "No route available."

        idx = self.state.get("current_waypoint_index", 0)
        while idx < len(path) - 1 and math.hypot(user_pos[0] - path[idx][0], user_pos[1] - path[idx][1]) < 5:
            idx += 1
        self.state["current_waypoint_index"] = idx

        if idx >= len(path) - 1:
            return "You have reached the exit."

        waypoint = path[idx]
        dz = waypoint[0] - user_pos[0]
        dx = waypoint[1] - user_pos[1]
        target_angle = math.degrees(math.atan2(dx, dz))
        delta = (target_angle - user_yaw + 540.0) % 360.0 - 180.0
        distance_m = math.hypot(dz, dx) * self.cell_size_m

        if abs(delta) <= 25:
            return f"Go straight for {distance_m:.1f} meters."
        if 25 < delta <= 120:
            return "Turn slightly right."
        if -120 <= delta < -25:
            return "Turn slightly left."
        return "Turn around."

    def stop_navigation(self):
        self.state.update({
            "mode": "idle",
            "active_goal": None,
            "active_path": [],
            "current_waypoint_index": 0,
            "navigation_target": None,
            "last_instruction": "",
            "previous_user_pos": None,
            "last_goal_distance_cells": None,
            "exit_reached": False,
        })
        return self.state

    def link_map_door(self, map_id, door_id, target_map_id, target_door_id):
        self.map_graph.setdefault(map_id, {"name": map_id, "exits": []})
        exits = self.map_graph[map_id].setdefault("exits", [])
        link = {
            "door_id": door_id,
            "target_map_id": target_map_id,
            "target_door_id": target_door_id,
        }
        exits[:] = [item for item in exits if item.get("door_id") != door_id]
        exits.append(link)
        self._save_map_graph()

        current_map = self.state.get("current_map")
        if current_map and current_map.get("map_id") == map_id:
            for landmark in current_map.get("landmarks", []):
                if landmark.get("id") == door_id:
                    landmark["target_map_id"] = target_map_id
                    landmark["target_door_id"] = target_door_id
            self.save_current_map()
        return link
