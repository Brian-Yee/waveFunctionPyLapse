#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from Distribution import Distribution
from IOTools import print_xml
from TileSet import TileSet


if __name__ == '__main__':
    # ts = TileSet('unittest/1.png', tile_dim=(16, 16))

    ts = TileSet('samples/tiled_images/Circuit.png', tile_dim=(14, 14))
    symmetries = [x.sym for x in ts.tile_dict.values()]

    dist = Distribution(ts)

    freq = dist.frequencies()
    neighbours = dist.allowed_neighbours(verbose=True)

    print_xml(symmetries, neighbours, freq)
    ts.save()
