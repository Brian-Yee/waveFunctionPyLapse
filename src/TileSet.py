#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import os
from IOTools import save_as_padded_rectangle, perceptual_sorter
from PIL import Image
from SymmetryTool import SymmetryTool
from subprocess import Popen
from Tile import Tile

from imagehash import average_hash

class TileSet:
    """Uniquely decomposition of an img into a set of permuted tiles."""
    ST = SymmetryTool()

    def __init__(self, fpath: str, tile_dim=(14, 14), dir_fpath='imgs'):
        img = self.read_in_img_as_array(fpath)
        tiles = self.tiles_from_img(img, tile_dim)

        self.tile_dict = self.define_basetiles(tiles)
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

    def define_basetiles(self, tiles: np.array, verbose=True) -> dict:
        """Define set of basetiles keyed by their hashes."""
        ravel_tiles = list(map(Tile, self.ravel_chunks(tiles)))
        hashes = np.array(list(map(hash, ravel_tiles)))

        _, unique_arg = np.unique(hashes, return_index=True)
        unique_tiles = (Tile(ravel_tiles[i].tile, self.ST) for i in unique_arg)
        basetiles= {hash(x): x for x in unique_tiles}

        if verbose:
            imgs = np.array([x.tile for x in basetiles.values()])
            N, dx, dy, dz = imgs.shape
            imgs = np.vstack(sorted(np.vsplit(imgs, N), key=perceptual_sorter))
            save_as_padded_rectangle('pngs/basetiles.png', imgs)

        return basetiles

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
