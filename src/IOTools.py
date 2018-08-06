#!/usr/bin/env python3
import numpy as np

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
