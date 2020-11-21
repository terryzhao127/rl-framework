import numpy as np
from io import BytesIO
from .data_pb2 import Data


def arr2bytes(arr):
    arr_bytes = BytesIO()
    np.save(arr_bytes, arr, allow_pickle=False)
    return arr_bytes.getvalue()


def bytes2arr(arr_bytes):
    arr = np.load(BytesIO(arr_bytes), allow_pickle=False)
    return arr
