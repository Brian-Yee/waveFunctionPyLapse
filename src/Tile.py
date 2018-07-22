#!/usr/bin/env python3
import numpy as np
import typing
np.set_printoptions(edgeitems=10)

class Tile:
    baseTile = np.array([])
    sym = None
    tol = 1e-5

    def __init__(self, A: np.array):
        self.baseTile = A
        self.sym = self.sym_group(A)

    def __repr__(self):
        msg = (
            self.baseTile.__repr__() + '\n'
            'Symmetry: {}'
        ).format(self.sym)

        return msg

    def __eq__(self, other) -> bool:
        """Check if a permutation of a tile is equivalent to another.

        First the possibility of equality from metadata is checked. Than
        equality is written as a series of booleans and generators so the
        minimum amount of computation is used to calculate equivalency.
        """
        if (type(other) is not type(self)) or (self.sym != other.sym):
            return False

        A, B = self.baseTile, other.baseTile
        return any(self.equal_under_rotation(A, X) for X in [B, np.fliplr(B)])

    def equal_under_rotation(self, A: np.array, B: np.array) -> bool:
        """Checks if any of the 4 rotations lead to array equality"""
        rotations = (np.rot90(B, k=k) for k in range(5))
        return any(np.allclose(A, X, atol=self.tol) for X in rotations)

    def sym_group(self, tile: np.array) -> str:
        """Checks for the symmetry group of a tile"""
        rot = [self.rotation(tile), self.rotation(np.rot90(tile))]
        sym = [self.symmetric(tile), self.symmetric(np.rot90(tile))]
        LR = [self.mirror(tile), self.mirror(np.rot90(tile))]

        if all(sym) and all(rot):
            if all(LR):
                return 'X'
            if not any(LR):
                return '\\'

        if not any(sym):
            if all(LR):
                return 'I'
            if any(LR) and not any(rot):
                return 'T'

        if any(sym) and not all(LR) and not all(rot):
            return 'L'

        return None

    def rotation(self, x: np.array) -> bool:
        return np.allclose(x, np.rot90(x, k=2), atol=self.tol)

    def symmetric(self, x: np.array) -> bool:
        return np.allclose(x, x.T, atol=self.tol)

    def mirror(self, x: np.array) -> bool:
        return np.allclose(x, np.fliplr(x), atol=self.tol)

    def combinations(self) -> list:
        A = self.baseTile
        return [[tuple(np.rot90(a, k=k).flatten()), (e, k)]
                for k in range(4, -1, -1)
                for e, a in enumerate([np.fliplr(A), A])]
