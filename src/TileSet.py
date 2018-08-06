#!/usr/bin/env python3
import numpy as np
from PIL import Image
from Tile import Tile
from SymmetryTool import SymmetryTool
import matplotlib.pyplot as plt
import sys
from subprocess import Popen
import os


class TileSet:
    """Uniquely decomposition of an img into a set of permuted tiles."""
    ST = SymmetryTool()

    def __init__(self, fpath: str, tile_dim=(14, 14), dir_fpath='imgs'):
        img = self.read_in_img_as_array(fpath)
        tiles = self.tiles_from_img(img, tile_dim)

        self.tile_dict = self.tile_dict(tiles, verbose=False)
        self.hmap = self.encode_basetiles(tiles)
        self.himg = self.encode_image(tiles, self.hmap)
        self.dir_fpath = dir_fpath

    @staticmethod
    def read_in_img_as_array(fpath):
        return np.array(Image.open(fpath))

    def tiles_from_img(self, img: np.array, tile_dim: list) -> np.array:
        """Chunk an image into equal array portions."""
        nrows, ncols = tile_dim
        h, w, d = img.shape
        chunks = img.reshape(h//nrows, nrows, -1, ncols, d)\
                    .swapaxes(1, 2)\
                    .reshape(h//nrows, w//ncols, nrows, ncols, d)
        return chunks

    def tile_dict(self, tiles: np.array, verbose=False):
        """Calculate tile_set with all permutations accounted for."""
        ravel_tiles = [Tile(x) for x in self.ravel_chunks(tiles)]
        hashed_tiles = np.array([hash(x) for x in ravel_tiles])
        hashes, args = np.unique(hashed_tiles, return_index=True)
        tile_dict = {hash(ravel_tiles[x]):
                     Tile(ravel_tiles[x].tile, self.ST) for x in args}

        if verbose:
            plt.imshow(np.hstack([x.tile for x in tile_dict.values()][:10]))
            plt.show()

        return tile_dict

    def encode_basetiles(self, tiles: np.array) -> dict:
        hmap = {}
        for _, tile in self.tile_dict.items():
            hmap = {**hmap, **tile.protocol_dict()}
        return hmap

    def encode_image(self, tiles: np.array, hmap: dict) -> np.array:
        ravel_tiles = [Tile(x) for x in self.ravel_chunks(tiles)]
        himg = np.array([hmap[x.flathash()] for x in ravel_tiles])\
                 .reshape([*tiles.shape[:2], 2])

        return himg

    @staticmethod
    def ravel_chunks(x):
        return x.reshape(-1, *x.shape[2:])

    def save(self):
        if not os.path.isdir(self.dir_fpath):
            Popen(['mkdir', self.dir_fpath]).wait()

        for e, (tile_hash, tile) in enumerate(self.tile_dict.items()):
            plt.imsave('{}/{:d}.png'.format(self.dir_fpath, int(tile_hash)), tile.tile)

    def load_tile(self, protocol):
        """Load tile based on hash and rotation."""
        tile_hash, k = protocol
        return self.tile_dict[tile_hash].rotate(k).tile



