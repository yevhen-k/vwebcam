import collections
import os
from threading import Event, Lock
from typing import Dict, Optional, Tuple
import json

__all__ = [
    'id2face_part',
    'StateChecker',
]

face_part = collections.namedtuple('parts', ['range', 'color'])
face_parts = {
    'face':       face_part(range(0, 17),   (173, 198, 231)),
    'eyebrow_r':  face_part(range(17, 22),  (0, 0, 255)),
    'eyebrow_l':  face_part(range(22, 27),  (0, 255, 0)),
    'nose':       face_part(range(27, 31),  (87,  60,  112)),
    'nostril':    face_part(range(31, 36),  (87,  60,  112)),
    'eye_r':      face_part(range(36, 42),  (0, 0, 125)),
    'eye_l':      face_part(range(42, 48),  (0, 125, 0)),
    'lips':       face_part(range(48, 60),  (151, 223, 137)),
    'teeth':      face_part(range(60, 68),  (151, 223, 137))
}


def id2face_part(idx: int) -> Tuple[Optional[Tuple[int, int, int]], Optional[str]]:
    for face_part_desc, part in face_parts.items():
        if idx in part.range:
            return part.color, face_part_desc
    return None, None


class StateChecker:
    def __init__(self, enabler_path: str, event: Event, lock: Lock, sleep_duration_sec: float) -> None:
        self.sleep_duration_sec = sleep_duration_sec
        self.enabler_path = enabler_path
        self.event = event
        self.lock = lock

        self.content_available = False
        self.event.clear()

        self.state = self.get_state()
        self.content = self.get_actual_content()

    def is_state_changed(self) -> bool:
        new_state = os.stat(self.enabler_path).st_mtime
        if new_state > self.state:
            return True
        return False

    def get_actual_content(self) -> Dict:
        with self.lock:
            with open(self.enabler_path, "rt") as f:
                content = json.load(fp=f)
            self.state = os.stat(self.enabler_path).st_mtime
            return content

    def get_state(self) -> float:
        return os.stat(self.enabler_path).st_mtime

    def enable_content_access(self) -> None:
        with self.lock:
            self.event.set()
            self.content_available = True

    def is_content_available(self) -> bool:
        with self.lock:
            return self.content_available

    def disable_content_access(self) -> None:
        with self.lock:
            self.event.clear()
            self.content_available = False
