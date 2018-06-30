#!/usr/bin/env python3
import numpy as np
import typing
np.set_printoptions(edgeitems=10)

class Tile:
    baseTile = np.array([])
    rotation = 0
    LR_flip = False
    sym = None
    tol = 1e-5

    def __init__(self, A: np.array):
        self.baseTile = A
        self.sym = self.sym_group(A)

    def __repr__(self):
        print(self.baseTile)

        msg = (
            'Rotation = {}\n'
            'LR_flip = {}\n'
            'Symmetry = {}'
        ).format(self.rotation, self.LR_flip, self.sym)

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
        LR = self.mirror(tile)
        if any(self.symmetric(np.rot90(tile, k=k)) for k in range(2)):
            return 'X' if LR else '/'
        else:
            TB = self.mirror(np.rot90(tile))
            if (LR and TB):
                return 'I'
            elif (not LR and TB) or (LR and not TB):
                return 'T'

        return 'L'

    def symmetric(self, x: np.array) -> bool:
        return np.allclose(x, x.T, atol=self.tol)

    def mirror(self, x: np.array) -> bool:
        return np.allclose(x, np.fliplr(x), atol=self.tol)

    def all_transforms(self) -> typing.Iterable[np.array]:
        """can prob del"""
        return (np.rot90(x, k=k) for k in range(4)
                for x in [self.baseTile, np.fliplr(self.baseTile)])

    def hashmap(self) -> dict:
        A = self.baseTile
        return {tuple(np.rot90(a, k=k).flatten()): (e, k)
                for k in range(4, -1, -1)
                for e, a in enumerate([np.fliplr(A), A])}
