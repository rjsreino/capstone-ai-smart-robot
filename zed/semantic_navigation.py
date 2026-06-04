import time
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")


class SemanticNavigator:
    def __init__(self):
        self.active_target = None
        self.active_target_class = None
        self.last_position = None
        self.last_depth = None
        self.last_seen_time = 0
        self.waiting_switch_candidate = None

    def find_best_match(self, user_command: str, detections: list[dict]):
        if not detections:
            return None

        detected_classes = []

        for d in detections:
            class_name = d.get("class_name") or d.get("class")
            if class_name:
                detected_classes.append(class_name)

        if not detected_classes:
            return None

        command_embedding = model.encode([user_command])
        class_embeddings = model.encode(detected_classes)

        similarities = cosine_similarity(command_embedding, class_embeddings)[0]
        best_index = int(np.argmax(similarities))
        best_score = float(similarities[best_index])

        if best_score < 0.25:
            return None

        best_class = detected_classes[best_index]

        best_detection = None
        best_depth = 9999

        for d in detections:
            class_name = d.get("class_name") or d.get("class")
            depth = d.get("depth_meters")

            if class_name == best_class:
                if depth is None:
                    depth = 9999

                if depth < best_depth:
                    best_depth = depth
                    best_detection = d

        return best_detection

    def start_navigation(self, user_command: str, detections: list[dict]):
        target = self.find_best_match(user_command, detections)

        if target is None:
            return "I cannot find a matching target right now."

        target_class = target.get("class_name") or target.get("class")
        position = target.get("position", "center")
        depth = target.get("depth_meters")

        self.active_target = target
        self.active_target_class = target_class
        self.last_position = position
        self.last_depth = depth
        self.last_seen_time = time.time()

        if depth is not None:
            return f"Guiding you to the {target_class}. It is on your {position}, about {depth:.1f} meters away."

        return f"Guiding you to the {target_class}. It is on your {position}."

    def update_navigation(self, detections: list[dict]):
        if self.active_target_class is None:
            return None

        current_candidates = []

        for d in detections:
            class_name = d.get("class_name") or d.get("class")

            if class_name == self.active_target_class:
                current_candidates.append(d)

        if not current_candidates:
            return f"I lost sight of the {self.active_target_class}. Stop and slowly scan left and right."

        current = min(
            current_candidates,
            key=lambda d: d.get("depth_meters") if d.get("depth_meters") is not None else 9999
        )

        position = current.get("position", "center")
        depth = current.get("depth_meters")

        if self.last_position == "left" and position == "right":
            self.last_position = position
            return f"You are moving away from the {self.active_target_class}. Turn back left."

        if self.last_position == "right" and position == "left":
            self.last_position = position
            return f"You are moving away from the {self.active_target_class}. Turn back right."

        if depth is not None and self.last_depth is not None:
            if depth > self.last_depth + 0.7:
                return f"You are moving away from the {self.active_target_class}. Turn back toward it."

        self.last_position = position
        self.last_depth = depth
        self.last_seen_time = time.time()

        closer_alt = self.find_closer_alternative(detections)

        if closer_alt:
            alt_class = closer_alt.get("class_name") or closer_alt.get("class")
            alt_pos = closer_alt.get("position", "center")
            alt_depth = closer_alt.get("depth_meters")

            self.waiting_switch_candidate = closer_alt

            if alt_depth is not None:
                return f"I found a closer {alt_class} on your {alt_pos}, about {alt_depth:.1f} meters away. Do you want to switch target?"

            return f"I found a closer {alt_class} on your {alt_pos}. Do you want to switch target?"

        if depth is not None and depth < 0.8:
            return f"You are very close to the {self.active_target_class}. Stop."

        if position == "left":
            direction = "Turn left."
        elif position == "right":
            direction = "Turn right."
        else:
            direction = "Move forward."

        if depth is not None:
            return f"{direction} The {self.active_target_class} is about {depth:.1f} meters away."

        return f"{direction} The {self.active_target_class} is ahead."

    def find_closer_alternative(self, detections: list[dict]):
        if self.last_depth is None:
            return None

        best_alt = None
        best_depth = self.last_depth

        for d in detections:
            class_name = d.get("class_name") or d.get("class")
            depth = d.get("depth_meters")

            if depth is None:
                continue

            if class_name == self.active_target_class:
                continue

            if depth + 0.5 < best_depth:
                best_depth = depth
                best_alt = d

        return best_alt

    def handle_switch_answer(self, command: str):
        command = command.lower()

        if self.waiting_switch_candidate is None:
            return None

        if "yes" in command or "switch" in command or "sure" in command:
            target = self.waiting_switch_candidate

            self.active_target = target
            self.active_target_class = target.get("class_name") or target.get("class")
            self.last_position = target.get("position", "center")
            self.last_depth = target.get("depth_meters")
            self.waiting_switch_candidate = None

            return f"Switching target to the {self.active_target_class}."

        if "no" in command or "keep" in command or "continue" in command:
            self.waiting_switch_candidate = None
            return f"Continuing to the {self.active_target_class}."

        return None

    def stop_navigation(self):
        self.active_target = None
        self.active_target_class = None
        self.last_position = None
        self.last_depth = None
        self.waiting_switch_candidate = None
        return "Navigation stopped."