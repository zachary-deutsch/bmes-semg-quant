"""Validated CPU interface around the attributed recovered QUANT implementation."""
import numpy as np
import torch

from ._quant import Quant


class QuantFeatures:
    def __init__(self, depth=2, div=4):
        if type(depth) is not int or not 1 <= depth <= 6:
            raise ValueError("depth must be an integer in [1, 6]")
        if type(div) is not int or div < 1:
            raise ValueError("div must be a positive integer")
        self.quant = Quant(depth=depth, div=div)

    @staticmethod
    def _tensor(x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3 or min(x.shape[:2]) < 1 or x.shape[-1] < 8 or not np.isfinite(x).all():
            raise ValueError("Expected finite nonempty examples x channels x length, length >= 8")
        return torch.from_numpy(np.ascontiguousarray(x))

    def fit_transform(self, x):
        tensor = self._tensor(x)
        self.shape = tensor.shape[1:]
        return self.quant.fit_transform(tensor, None).numpy()

    def transform(self, x):
        tensor = self._tensor(x)
        if not hasattr(self, "shape"):
            raise ValueError("QUANT is not fitted")
        if tensor.shape[1:] != self.shape:
            raise ValueError("Transform channels and length must match training")
        return self.quant.transform(tensor).numpy()
