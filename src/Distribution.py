#!/usr/bin/env python3
import numpy as np
from TileSet import TileSet
import matplotlib.pyplot as plt


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
        pairs = np.vstack([self.vertical_pairs(self.tileset.himg),
                           self.horizontal_pairs(self.tileset.himg)])

        neighbours = self.unique_basetile_pairs(pairs)

        if verbose:
            self.visualize_neighbours(neighbours)

        raise SystemExit
        return neighbours

    @staticmethod
    def horizontal_pairs(x: np.array) -> np.array:
        """Create array of horizontal adjacent pairs in matrix."""
        left, right = x[:, :-1, :], x[:, 1:, :]
        return np.dstack([left, right]).reshape(-1, 2, 2)

    @staticmethod
    def vertical_pairs(x: np.array) -> np.array:
        """Create array of horizontal adjacent pairs in matrix."""
        # change protocol to rotate image such that top bottom pairs
        # become left right pairs indexed on top bottom spots
        x[:, :, 1] = (x[:, :, 1] + 1) % 4

        top, bottom = x[:-1, :, :], x[1:, :, :]
        return np.dstack([top, bottom]).reshape(-1, 2, 2)

    @staticmethod
    def unique_basetile_pairs(pairs: np.array) -> np.array:
        """Return unique basetile pairs assuming followed protocol."""
        basetile_info_arg = 0
        basetile_pairs = pairs[:, :, basetile_info_arg].reshape(-1, 2)
        _, args = np.unique(basetile_pairs, axis=0, return_index=True)

        return np.array([pairs[x] for x in args])

    def visualize_neighbours(self, neighbours):
        """Create image for visual verification of calculated neighbours."""
        w = self.largest_squarewidth_possible(neighbours)
        img = self.construct_img(neighbours, w)
        img = self.add_visual_grid(img)

        plt.imshow(img)
        plt.show()

    @staticmethod
    def largest_squarewidth_possible(neighbours):
        return int(np.floor(np.sqrt(neighbours.shape[0])))

    def construct_img(self, neighbours, w):
        """Constructs minimal square image of neighbouring pairs."""
        pair_imgs = np.array([self.create_pair_img(p) for p in neighbours[:w**2]])
        vertical_strips = [np.vstack(x) for x in np.split(pair_imgs, w)]
        return np.hstack(vertical_strips)

    def create_pair_img(self, p):
        return np.hstack([self.tileset.load_tile(x) for x in p])

    @staticmethod
    def add_visual_grid(img):
        # draw visual row seperators
        for x in range(14, img.shape[0], 14):
            img[x:x+1, :] = 1

        # draw visual column seperators
        for x in range(2*14, img.shape[1], 2*14):
            img[:, x:x+1] = 1

        return img

