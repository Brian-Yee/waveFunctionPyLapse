#!/usr/bin/env python3
import numpy as np
from PIL import Image
from Tile import Tile
import matplotlib.pyplot as plt


class TileSet:
    """Uniquely decomposition of an img into a set of permuted tiles."""
    def __init__(self, fpath: str, tile_dim=(14, 14)):
        # img = np.array(Image.open(fpath).convert('L'))
        img = np.array(Image.open(fpath))

        self.tile_dim = tile_dim
        self.img_dim= img.shape

        observations = self.tiles_from_img(img)
        self.tile_set = self.tile_set(observations)
        self.hmap = self.hashmap(self.tile_set)
        rev_hmap = {v: k for k, v in self.hmap.items()}
        self.himg= self.hash_observations_as_img(observations, rev_hmap)

    def __repr__(self):
        msg = (
            str(self.spatial_tile_set.__repr__()) + '\n'
            'tile set size: {}\n'
            'hmap size    : {}'
        ).format(len(self.tile_set), len(self.hmap))
        return msg

    def tiles_from_img(self, img:np.array) -> np.array:
        """Chunk an image into equal array portions."""
        (h, w, d), (nrows, ncols) = self.img_dim, self.tile_dim
        chunks = img.reshape(h//nrows, nrows, -1, ncols, d)\
                    .swapaxes(1, 2)\
                    .reshape(-1, nrows, ncols, d)
        return chunks

    @staticmethod
    def tile_set(observations: np.array, verbose=False):
        """Calculate tile_set with all permutations accounted for."""
        tile_set = []
        for x in map(Tile, np.unique(observations, axis=0)):
            if x not in tile_set:
                tile_set.append(x)

        if verbose:
            plt.imshow(np.hstack([x.baseTile for x in tile_set]))
            plt.show()

        return tile_set

    @staticmethod
    def hashmap(tile_set):
        """Return the permutations of tile_set to recreate image."""
        hmap = {}
        for e, tile in enumerate(tile_set):
            hmap_0 = {tuple([e, *v]): k for k, v in tile.combinations()}
            hmap = {**hmap, **hmap_0}
        return hmap

    def hash_observations_as_img(self, observations, hmap):
        # map each tile image to it's orientation in term of true north tiles
        tile_lengths = (self.img_dim[e]//self.tile_dim[e] for e in range(2))
        flat_hash = [hmap[tuple(x.flatten())] for x in observations]
        return np.array(flat_hash).reshape(*tile_lengths, 3)
