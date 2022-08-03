import collections
from typing import Optional, Tuple

__all__ = [
    'id2face_part',
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
