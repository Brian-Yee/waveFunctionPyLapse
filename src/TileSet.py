#!/usr/bin/env python3
from Tile import Tile
import numpy as np
import matplotlib.pyplot as plt
import re
from Tile import Tile
from glob import glob
from pprint import pprint
from PIL import Image


class TileSet:
    def __init__(self, fpath: str, tile_dim=(14, 14)):
        img = np.array(Image.open(fpath).convert('L'))

        self.tile_dim = tile_dim
        self.img_dim= img.shape

        observations = self.tiles_from_img(img)
        tile_set = self.tile_set(observations)
        hmap = self.hashmap(tile_set)
        img_tt = self.hash_observations(observations, hmap)

        print(img_tt)

    def tiles_from_img(self, img:np.array) -> np.array:
        """Chunk an image into equal array portions."""
        (h, w), (nrows, ncols) = self.img_dim, self.tile_dim

        return img.reshape(h//nrows, nrows, -1, ncols)\
                  .swapaxes(1, 2)\
                  .reshape(-1, nrows, ncols)

    @staticmethod
    def tile_set(observations: np.array):
        """Calculate tile_set with all permutations accounted for."""
        tile_set = []
        for x in map(Tile, np.unique(observations, axis=0)):
            if x not in tile_set:
                tile_set.append(x)

        return tile_set

    @staticmethod
    def hashmap(tile_set):
        """Return the permutations of tile_set to recreate image."""
        hmap = {}
        for e, tile in enumerate(tile_set):
            hmap_0 = {k: [e, *v] for k, v in tile.hashmap().items()}
            hmap = {**hmap, **hmap_0}
        return hmap

    def hash_observations(self, observations, hmap):
        # map each tile image to it's orientation in term of true north tiles
        tile_lengths = (self.img_dim[e]//self.tile_dim[e] for e in range(2))
        flat_hash = [hmap[tuple(x.flatten())] for x in observations]
        return np.array(flat_hash).reshape(*tile_lengths, 3)

    def unique_tiles(self, tiles:np.array) -> np.array:
        return np.vstack({self.as_tuple(tile) for tile in tiles})\
                 .reshape(-1, *self.tile_dim)

    def reduce_symmetries(self, tiles:np.array) -> set:
        tile_set = set()
        for tile in tiles:
            tile_transformations = map(self.as_tuple, self.all_transforms(tile))
            if all(x not in tile_set for x in tile_transformations):
                tile_set.update([self.as_tuple(tile)])
        return tile_set

    def create_symmetries(self, tiles:set) -> set:
        """XXX: come back to here basically I am trying to get a way of indexing the original
        image to see how the tiles came about so there are 8 possible transforms from 4 90 rots
        and 2 lr flips and I want to reverse engineer dependencies from the image knowing this

        Data structure to stole all tiles
        True north
        Four rotations
        LR Flip boolean
        Symmetry Group
        """
        tile_set = {}
        for tile in tiles:
            tile_transformations = map(self.as_tuple, self.all_transforms(self.as_tile(tile)))
            tile_set = {**tile_set, **{x: tile for x in tile_transformations}}

        return tile_set

    @staticmethod
    def as_tuple(x):
        return tuple(x.flatten())

    def as_tile(self, x):
        return np.array(x).reshape(*self.tile_dim)
