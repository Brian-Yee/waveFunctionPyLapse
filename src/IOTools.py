#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from imagehash import average_hash
from PIL import Image

def save_as_padded_rectangle(fpath, imgs, pad=3):
    # add leftover padding and visual seperator whitespace
    N, dx, dy, dz = imgs.shape
    w = int(np.ceil(np.sqrt(N)))
    ndim_pad = ((0, w**2 - N), (0, pad), (0, pad), (0, 0))

    # rearrange image to rectangle with padding
    img = np.pad(imgs, ndim_pad, 'constant')\
            .reshape(w, w, dx+pad, dy+pad, dz)\
            .swapaxes(0, 1)\
            .swapaxes(1, 2)\
            .reshape(w*(dx+pad), w*(dy+pad), -1)

    plt.imsave(fpath, img)
    plt.show()


def perceptual_sorter(x, slice_func=None):
    arr = x.squeeze(axis=0)
    if slice_func is not None:
        arr = slice_func(arr)

    img = Image.fromarray(arr)
    return int('0x' + str(average_hash(img)), 0)


def print_xml(symmetries: np.array,
              neighbours: np.array,
              freqs:np.array,
              verbose=True):
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
