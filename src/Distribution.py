#!/usr/bin/env python3
import numpy as np
from TileSet import TileSet
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self, tileset: TileSet):
        self.tileset = tileset

    def frequencies(self, normalized=True):
        """Calculate frequency basetiles."""
        himg = self.tileset.himg
        base_tiles, _ = np.split(himg, himg.shape[-1], axis=-1)
        freq = np.dstack(np.unique(base_tiles.flatten(), return_counts=True))\
                .squeeze()

        return freq

    def allowed_neighbours(self, verbose=False):
        """Calculate left right neighbours."""
        # TODO add top bottom pairs later as well
        himg = self.tileset.himg
        lr = np.dstack([himg[:, :-1, :], himg[:, 1:, :]])\
               .reshape(-1, 2, 2)

        _, args = np.unique(lr[:, :, 0].reshape(-1, 2), axis=0,
                            return_index=True)
        neighbours = np.array([lr[x] for x in args])\
                       .reshape(-1, 2, 2)
        if verbose:
            W = int(np.floor(np.sqrt(neighbours.shape[0])))
            pair_imgs = [self.create_pair_img(p) for p in neighbours[:W**2]]
            img = np.hstack([np.vstack(pair_imgs[W*x:W*(x+1)]) for x in range(W)])

            # insert vertical lines
            for x in range(0, img.shape[1], 2*14):
                img[:, x:x+1] = 1

            plt.imshow(img)
            plt.show()

        return neighbours

    def create_pair_img(self, p):
        return np.hstack([self.tileset.load_tile(x) for x in p])
