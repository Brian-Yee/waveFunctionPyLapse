#!/usr/bin/env python3
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpl_img
from pprint import pprint
import re


def read_in_tile_set(dir_path):
    """Reads in all images with corresponding names and derives symmetry."""
    tile_fpaths = glob('{}/*.png'.format(dir_path))
    name = lambda x: re.search('{}/(\w+).png'.format(dir_path), x).group(1)
    tileset = {name(fpath): mpl_img.imread(fpath) for fpath in tile_fpaths}

    # infer symmetries of tiles
    sym_groups = {k: sym_group(v) for k, v in tileset.items()}

    pprint(sym_groups)

def sym_group(tile, tol=1e-4):
    """Checks for the symmetry group of a tile"""
    symmetric = lambda x: np.allclose(x, x.T, atol=tol)
    mirror = lambda x: np.allclose(x, np.fliplr(x), atol=tol)

    tile = np.mean(tile, axis=-1)
    if symmetric(tile):
        return 'X' if mirror(tile) else '/'
    elif mirror(tile):
        return 'I' if mirror(np.rot90(tile)) else 'T'
    else:
        return 'L'

if __name__ == '__main__':
    read_in_tile_set('samples/Circuit')
