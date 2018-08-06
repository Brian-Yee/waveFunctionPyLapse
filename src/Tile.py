#!/usr/bin/env python3
import numpy as np
import typing
import sys
from itertools import product
import ctypes
import matplotlib.pyplot as plt

class SymmetryIdentifier:
    tol = 1e-5

    def identify(self, tile: np.array) -> str:
        return self.hmap[self.truthtable(tile)]

    def __init__(self):
        self.hmap = self.truthtable_hashmap()

    def truthtable_hashmap(self) -> dict:
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


class Tile:
    tol = 1e-5

    def __init__(self, tile: np.array, SI: SymmetryIdentifier=None):
        self.tile = tile

        if SI is not None:
            self.sym = SI.identify(tile)
        else:
            self.sym = None

    def __repr__(self):
        return 'Hash: {}, Symmetry: {}'.format(self.__hash__(), self.sym)

    def __hash__(self):
        """Quick and dirty hash to return only positive values."""
        sorted_pixel_channels = tuple(np.sort(self.tile.reshape(-1)))
        return hash(sorted_pixel_channels)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not(self == other)

    def combinations(self) -> dict:
        """Create all possile combinations of a tile."""
        return {self.transform(k).flathash(): (hash(self), k) for k in range(4)}

    def transform(self, rotation):
        # TODO retain symmetry
        array = np.rot90(self.tile, k=rotation)
        return Tile(array)

    def flathash(self) -> int:
        """Quick and dirty hash to return only positive values."""
        return hash(tuple(self.tile.flatten()))
