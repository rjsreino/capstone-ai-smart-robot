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

DYNAMIC_CLASSES = {
    "person",
    "dog",
    "cat",
    "chair",
    "backpack",
    "handbag",
    "suitcase",
    "bottle",
    "cup",
    "box",
}

EXIT_LANDMARK_TYPES = {"door", "doorway", "exit sign"}


def _clamp(value, low, high):
    return max(low, min(value, high))


def _normalize_label(label):
    return str(label or "").strip().lower().replace("_", " ").replace("-", " ")


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

    def pose_to_grid(self, pose):
        x_m = float(pose.get("x", 0.0)) / 1000.0
        z_m = float(pose.get("z", 0.0)) / 1000.0
        gx = _clamp(int(x_m / self.cell_size_m) + self.width // 2, 0, self.width - 1)
        gz = _clamp(int(z_m / self.cell_size_m) + self.height // 2, 0, self.height - 1)
        return gz, gx

    def _update_user_pose(self, pose):
        self.state["user_pose"] = {
            "x": float(pose.get("x", 0.0)) / 1000.0,
            "z": float(pose.get("z", 0.0)) / 1000.0,
            "yaw": float(pose.get("yaw", 0.0)),
        }

    def _create_empty_map(self, map_name, static_grid=None, map_id=None):
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
            "metadata": {
                "scan_quality": 0.0,
                "coverage_percent": 0.0,
            },
        }

    def _grid_coverage(self, grid):
        total = self.width * self.height
        known = sum(1 for row in grid for cell in row if int(cell) != 2)
        return round((known / total) * 100.0, 2)

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
            if label not in DYNAMIC_CLASSES:
                continue
            gx = obj.get("x")
            gz = obj.get("z")
            if gx is None or gz is None:
                continue
            obstacles.append({
                "class_name": label,
                "grid_x": _clamp(int(gx), 0, self.width - 1),
                "grid_z": _clamp(int(gz), 0, self.height - 1),
                "distance": obj.get("distance"),
            })
        self.state["live_dynamic_obstacles"] = obstacles
        return obstacles

    def start_mapping(self, map_name, live_grid, semantic_objects, pose, map_id=None):
        current_map = self._create_empty_map(map_name, live_grid, map_id=map_id)
        self.update_static_landmarks(current_map, semantic_objects)
        current_map["metadata"]["coverage_percent"] = self._grid_coverage(current_map["static_grid"])
        current_map["metadata"]["scan_quality"] = min(1.0, current_map["metadata"]["coverage_percent"] / 85.0)
        self.state.update({
            "mode": "mapping",
            "current_map_id": current_map["map_id"],
            "current_map": current_map,
            "active_goal": None,
            "active_path": [],
            "current_waypoint_index": 0,
        })
        self._update_user_pose(pose)
        return current_map

    def refresh_mapping(self, live_grid, semantic_objects):
        current_map = self.state.get("current_map")
        if not current_map:
            return None
        current_map["static_grid"] = self._normalize_grid(live_grid)
        self.update_static_landmarks(current_map, semantic_objects)
        current_map["metadata"]["coverage_percent"] = self._grid_coverage(current_map["static_grid"])
        current_map["metadata"]["scan_quality"] = min(1.0, current_map["metadata"]["coverage_percent"] / 85.0)
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
        for path in sorted(self.maps_dir.glob("*.json")):
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
                "landmark_count": len(data.get("landmarks", [])),
                "coverage_percent": data.get("metadata", {}).get("coverage_percent", 0.0),
            })
        return maps

    def load_map(self, map_id=None, map_name=None):
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
            self.state["current_map_id"] = data.get("map_id")
            self.state["current_map"] = data
            return data
        return None

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
        current_map = self.state.get("current_map") or self.load_map()
        if not current_map:
            return None, "No saved map loaded."

        self._update_user_pose(pose)
        user_pos = self.pose_to_grid(pose)
        live_obstacles = self.get_live_dynamic_obstacles(semantic_objects)
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
        live_obstacles = self.get_live_dynamic_obstacles(semantic_objects)
        nav_grid = self.build_navigation_grid(self.state["current_map"]["static_grid"], live_obstacles)
        rerouted = False
        warning = None

        if self.detect_wrong_direction(user_pos):
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
        self.state["last_instruction"] = instruction
        return self._guidance_response(instruction, rerouted=rerouted)

    def _guidance_response(self, instruction, rerouted=False):
        path = self.state.get("active_path", [])
        idx = self.state.get("current_waypoint_index", 0)
        remaining_cells = max(0, len(path) - idx - 1)
        target = self.state.get("navigation_target") or {}
        return {
            "active": True,
            "instruction": instruction,
            "target": target.get("id"),
            "remaining_distance_m": round(remaining_cells * self.cell_size_m, 2),
            "rerouted": rerouted,
            "path": path,
            "goal": {"z": self.state["active_goal"][0], "x": self.state["active_goal"][1]},
        }

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
