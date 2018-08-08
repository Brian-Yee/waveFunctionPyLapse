#!/usr/bin/env python3
import numpy as np
from SymmetryTool import SymmetryTool
# import imagehash


class Tile:
    tol = 1e-5

    def __init__(self, tile: np.array, ST: SymmetryTool=None):
        self.sym = ST.identify(tile) if ST is not None else None
        self.tile = tile

    def __repr__(self):
        return 'Hash: {}, Symmetry: {}'.format(self.__hash__(), self.sym)

    def __hash__(self):
        """Quick and dirty translationally flat invariant hash."""
        # fingerprint = tuple(np.sort(self.tile.reshape(-1)))
        fingerprint = tuple(self.tile.reshape(-1))
        return hash(fingerprint)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __ne__(self, other):
        return not(self == other)

    def protocol_dict(self) -> dict:
        """Relate an observed tile record with a protocol for reconstruction."""
        # TODO: implement flipping as part of the legend
        # protocol = {self.flip(f).rotate(k).flathash(): (hash(self), k, f)
        #             for k in range(4)
        #             for f in range(2)}
        protocol = {self.rotate(k).flathash(): (hash(self), k)
                    for k in range(4)}
        return protocol

    def rotate(self, rotation):
        """Return a new instance of an object with a rotated tile."""
        tile = Tile(np.rot90(self.tile, k=rotation))
        tile.sym = self.sym
        return tile

    def flip(self, flip):
        """Return a new instance of an object with a rotated tile."""
        if flip % 2 == 1:
            tile = Tile(np.fliplr(self.tile))
            tile.sym = self.sym
        else:
            tile = self

        return tile


    def flathash(self) -> int:
        """Return tile to be hashed as is."""
        return hash(tuple(self.tile.flatten()))
