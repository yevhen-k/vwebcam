import collections
from typing import Tuple
import numpy as np


pred_type = collections.namedtuple('prediction_type', ['range', 'color'])
pred_types = {'face':       pred_type(range(0, 17),   (173, 198, 231)),
              'eyebrow_r':  pred_type(range(17, 22),  (0, 0, 255)),
              'eyebrow_l':  pred_type(range(22, 27),  (0, 255, 0)),
              'nose':       pred_type(range(27, 31),  (87,  60,  112)),
              'nostril':    pred_type(range(31, 36),  (87,  60,  112)),
              'eye_r':      pred_type(range(36, 42),  (0, 0, 125)),
              'eye_l':      pred_type(range(42, 48),  (0, 125, 0)),
              'lips':       pred_type(range(48, 60),  (151, 223, 137)),
              'teeth':      pred_type(range(60, 68),  (151, 223, 137))
              }

def color_by_id(idx: int) -> Tuple[int, int, int]:
    for key, preds in pred_types.items():
        if idx in pred_types[key].range:
            return pred_types[key].color
    return (0, 0, 0)