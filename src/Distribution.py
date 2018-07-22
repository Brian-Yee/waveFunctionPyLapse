#!/usr/bin/env python3
import numpy as np
from TileSet import TileSet
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self, tile_set: TileSet):
        self.ts = tile_set.tile_set
        self.himg = tile_set.himg
        self.hmap = tile_set.hmap

    def frequencies(self):
        """Calculate frequency of tile tuples."""
        freq = np.bincount(self.himg.reshape(-1, 3)[:, 0].flatten())
        return freq/freq.sum()

    def allowed_neighbours(self, verbose=False):
        """Calculate left right neighbours."""
        # TODO add top bottom pairs later as well
        l, r = self.himg[:-1, :, :], self.himg[1:, :, :]

        lr_pairs = np.vstack([x.reshape(-1, 3) for x in [l, r]])\
                     .reshape(-1, 2, 3)

        unique_pairs = np.unique(lr_pairs.reshape(-1, 6), axis=0)\
                         .reshape(-1, 2, 3)

        if verbose:
            hash_to_tile = lambda x: np.array(self.hmap[tuple(x)]).reshape(14, 14)
            pairs = [np.hstack(list(map(hash_to_tile, p))) for p in unique_pairs]

            width = int(np.ceil(np.sqrt(len(pairs))))

            # apd the future visual image
            for x in range(len(pairs), width**2):
                pairs.append(np.zeros_like(pairs[0]))

            visual = np.array(pairs).reshape(width, width*14, 28)
            visual = np.vstack(np.hstack(visual))

            for x in range(0, visual.shape[1], 28):
                visual[:, x] = 0

            plt.imshow(np.vstack(visual))
            plt.show()

        return unique_pairs

    def __repr__(self):
        return 'NONE'
