#!/usr/bin/env python3
from TileSet import TileSet
from Distribution import Distribution
import numpy as np
import matplotlib.pyplot as plt


def print_xml(symmetries, freqs, verbose=True):
    """Add in a conditional space for the name."""
    # NOTE: this is a quick hack to see if I can get something working
    # in reality we need to ensure that all tiles are based off of the ground
    # tile permutation

    tuples = sorted(zip(freqs, symmetries), key=lambda x: x[1])
    print('<set size="14">')
    print('\t<tiles>')
    for (name, freq), sym in tuples:
        print('\t\t<tile name="{:d}" symmetry="{:s}" weight="{:d}"/>'.format(name, sym, freq))
    print('\t</tiles>')

    print('\t<neighbors>')
    for n in neighbours:
        name = n[:, 0]
        rot = n[:, 1]
        item = '\t\t<neighbor left="{} {}" right="{} {}"/>'.format(
            name[0], rot[0], name[1], rot[1])
        print(item)
    print('\t</neighbors>')

    print('</set>')

if __name__ == '__main__':
    ts = TileSet('samples/tiled_images/Circuit.png')
    dist = Distribution(ts)

    freq = dist.frequencies()
    neighbours = dist.allowed_neighbours(verbose=True)

    symmetries = [x.sym for x in dist.tileset.tile_set]
    print_xml(symmetries, freq)
