#!/usr/bin/env python3
import numpy as np
from itertools import product


class SymmetryTool:
    tol = 1e-5

    def identify(self, tile: np.array) -> str:
        """Identify the symmetry group of a tile."""
        return self.hashmap[self.truthtable(tile)]

    def __init__(self):
        self.hashmap = self.truthtable_symmetries()

    def truthtable_symmetries(self) -> dict:
        """Derive symmetry type from propeties of simple representations."""
        examples = {
            'X': np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]]),
            'I': np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]]),
            'T': np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]]),
            'L': np.array([[0, 1, 0], [0, 1, 1], [0, 0, 0]]),
            '\\': np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])}

        hmap = {self.truthtable(np.rot90(img, k=k)): sym
                for k, (sym, img) in product(range(4), examples.items())}

        return hmap

    def truthtable(self, x: np.array) -> tuple:
        """Combine all matrix propertes of image."""
        truthtable = [self.rotation(x), self.symmetric(x),
                      *[self.mirror(np.rot90(x, k=k)) for k in range(2)]]

        return tuple(truthtable)

    def mirror(self, x: np.array) -> bool:
        """Check for mirror symmetric matrix propetry."""
        return np.allclose(x, np.fliplr(x), atol=self.tol)

    def rotation(self, x: np.array) -> bool:
        """Check for rotational invariance of matrix for 180\deg."""
        return np.allclose(x, np.rot90(x, k=2), atol=self.tol)

    def symmetric(self, x: np.array) -> bool:
        """Check for a symmetric matrix property across all planes."""
        if x.ndim == 2:
            x = np.expand_dims(x, axis=-1)
        return all(map(self.symmetric_2d, np.rollaxis(x, axis=-1)))

    def symmetric_2d(self, x: np.array) -> bool:
        """Check for a symmetric matrix property across one plane."""
        return np.allclose(x, x.T, atol=self.tol)

