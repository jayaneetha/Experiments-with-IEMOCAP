from hashlib import sha1

import numpy as np


def get_hash(arr: np.ndarray) -> str:
    x = np.ascontiguousarray(arr)
    s = sha1(x)
    return s.hexdigest()
