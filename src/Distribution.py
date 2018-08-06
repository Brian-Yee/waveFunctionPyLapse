#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from TileSet import TileSet


class Distribution:
    def __init__(self, tileset: TileSet):
        self.tileset = tileset

    def frequencies(self, normalized=True):
        """Calculate frequency basetiles."""
        protocols = self.tileset.himg.shape[-1]
        basetiles, _ = np.split(self.tileset.himg, protocols, axis=-1)
        freq = np.dstack(np.unique(basetiles.flatten(), return_counts=True))\
                .squeeze()

        return freq

    def allowed_neighbours(self, verbose=False):
        """Calculate all observed possible neighbour pairings."""
        # TODO: two very odd allowances are allowed when combining
        #       vertical and horizontal pairs but not when done
        #       seperately. Figure out why eventually

        pairs = np.vstack([self.horizontal_pairs(self.tileset.himg)])
        neighbours = self.unique_basetile_pairs(pairs)

        if verbose:
            self.visualize_neighbours(neighbours)

        return neighbours

    @staticmethod
    def horizontal_pairs(x: np.array) -> np.array:
        """Create array of horizontal adjacent pairs in matrix."""
        left, right = x[:, :-1, :], x[:, 1:, :]
        return np.dstack([left, right]).reshape(-1, 2, 2)

    @staticmethod
    def vertical_pairs(x: np.array) -> np.array:
        """Create array of vertical adjacent pairs in matrix.

        For efficient calculate one needs to modify the protocols such
        that top bottom pairs become left right pairs indexed on top
        bottom spots. This can be quite confusing so to explain it
        put your hands out in front of you joining your index finger
        and thumb with all knuckles pointing RIGHT. Keeping your wrists
        stationary, rotate your left hand 90 degrees COUNTER-CLOCKWISE^1
        and then rotate your right hand 90 degress COUNTER-CLOCKWISE.
        All your knuckles should now be pointing UPWARDS. Recall originally
        your index and thumb were originally joined. In this new
        reference space while your wrists (array locations) are top/bottom
        the image of your hands are now now left/right.

        [1] For additional persuastion print
            x vs np.rot90(x)
            to see that it is a counter-clockwise rotation.
        """
        x[:, :, 1] = (x[:, :, 1] + 1) % 4

        top, bottom = x[:-1, :, :], x[1:, :, :]
        return np.dstack([top, bottom]).reshape(-1, 2, 2)

    @staticmethod
    def unique_basetile_pairs(pairs: np.array) -> np.array:
        """Return unique basetile pairs assuming followed protocol."""
        basetile_info_arg = 0
        basetile_pairs = pairs[:, :, basetile_info_arg].reshape(-1, 2)
        _, args = np.unique(np.sort(basetile_pairs), axis=0, return_index=True)

        return np.array([pairs[x] for x in args])

    def visualize_neighbours(self, neighbours, pad=3):
        """Create image for visual verification of calculated neighbours.

        To visualize all tiles reasonably on a screen we rearrange tiles
        side by side with some padding between in the shape of a
        rectangle of dimensions [w, 2*w]
        """
        sorted_neighbours = sorted(neighbours, key=lambda x: x[0][0])
        img_pairs = np.array([self.create_pair_img(x)
                              for x in sorted_neighbours])

        # add leftover padding and visual seperator whitespace
        N, dx, dy, dz = img_pairs.shape
        w = int(np.ceil(np.sqrt(N)))
        ndim_pad = ((0, w**2 - N), (0, pad), (0, pad), (0, 0))

        # rearrange image to rectangle with padding
        img = np.pad(img_pairs, ndim_pad, 'constant')\
                .reshape(w, w, dx+pad, dy+pad, dz)\
                .swapaxes(1, 2)\
                .reshape(w*(dx+pad), w*(dy+pad), -1)

        plt.imshow(img)
        plt.show()

    def create_pair_img(self, p):
        return np.hstack([self.tileset.load_tile(x) for x in p])
