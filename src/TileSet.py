#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from SymmetryTool import SymmetryTool
from subprocess import Popen
from Tile import Tile


class TileSet:
    """Uniquely decomposition of an img into a set of permuted tiles."""
    ST = SymmetryTool()

    def __init__(self, fpath: str, tile_dim=(14, 14), dir_fpath='imgs'):
        img = self.read_in_img_as_array(fpath)
        tiles = self.tiles_from_img(img, tile_dim)

        self.tile_dict = self.tile_dict(tiles, verbose=False)
        self.hmap = self.define_protocol(tiles)
        self.himg = self.apply_protocol(tiles, self.hmap)
        self.dir_fpath = dir_fpath

    @staticmethod
    def read_in_img_as_array(fpath: str) -> np.array:
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
        """Define set of basetiles keyed by their hashes."""
        ravel_tiles = list(map(Tile, self.ravel_chunks(tiles)))
        hashes = np.array([hash(x) for x in ravel_tiles])
        _, args = np.unique(hashes, return_index=True)
        tiles = [ravel_tiles[x] for x in args]

        tile_dict = {hash(x): Tile(x.tile, self.ST) for x in tiles}

        if verbose:
            plt.imshow(np.hstack([x.tile for x in tile_dict.values()]))
            plt.show()

        return tile_dict

    def define_protocol(self, tiles: np.array) -> dict:
        """Add all protocols to one dictionary.

        Keys: flathashes
        Valuse: [hash(basetile), turns]

        flathash := rotate(hash(basetile), turns)
        """
        hmap = {}
        for _, tile in self.tile_dict.items():
            hmap = {**hmap, **tile.protocol_dict()}
        return hmap

    def apply_protocol(self, tiles: np.array, hmap: dict) -> np.array:
        ravel_tiles = map(Tile, self.ravel_chunks(tiles))
        himg = np.array([hmap[x.flathash()] for x in ravel_tiles])\
                 .reshape([*tiles.shape[:2], 2])

        return himg

    @staticmethod
    def ravel_chunks(x: np.array) -> np.array:
        return x.reshape(-1, *x.shape[2:])

    def save(self) -> None:
        """Save basetile images with hashes as filename."""
        if not os.path.isdir(self.dir_fpath):
            Popen(['mkdir', self.dir_fpath]).wait()

        for e, (tile_hash, tile) in enumerate(self.tile_dict.items()):
            plt.imsave('{}/{:d}.png'.format(self.dir_fpath, int(tile_hash)), tile.tile)

    def load_tile(self, protocol: np.array) -> np.array:
        """Load tile based on hash and rotation."""
        tile_hash, k = protocol
        return self.tile_dict[tile_hash].rotate(k).tile



