#!/usr/bin/env python3
from TileSet import TileSet
from Distribution import Distribution
from IOTools import print_xml
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    ts = TileSet('samples/tiled_images/Circuit.png', tile_dim=(14, 14))
    symmetries = [x.sym for x in ts.tile_dict.values()]

    dist = Distribution(ts)

    freq = dist.frequencies()
    neighbours = dist.allowed_neighbours(verbose=True)

    print_xml(symmetries, neighbours, freq)
    ts.save()
