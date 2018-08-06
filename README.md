# waveFunctionPyLapse
The goal of this project is to build upon the work of waveFunctionCollapse to generate new maps from previously tiled map inputs

# TODO
- Dockerize Fehrant's version of waveFunctionCollapse
- Test image decomposition
- Implement picture hashing for slightly imperfect tiled maps

# Method
1. Read in an image
2. Chunk the image to a predefined tile size
3. Build a tile set that can reconstruct the image
- Infer all symmetries of the tiles
- Record all observed legal neighbour pairings
4. Write `xml` input for wavefunction collapse

The tricky parts come in at 3.a and 3.b here's how they are managed accordingly

## 3.a)
A dictionary keyed by a truthtable defining how a tile equates to itself under transformation is used to identify a symmetry value

## 3.b)
A dictionary keyed by a flattened hash (`flathash`) of a tile leads to a list composed of value protocols. A protocol is defined to be a (`hash`, `rotation`) to create the dictionary key's hash. A hashing a `basetile` denoted by `hash` under a `rotation` and applying `flathash` recovers the original key. A `basetile` is defined as the first observed unique tile when reading in an image's tiles lexicographically. A tile is said to `hash` to a `basetile` if the `hash` of all it's pixels *when sorted* is equivalent to the sorted `basetile`s sorted pixels. Note collisions can occur under two conditions of this assumtion:

1. A subset of the image is rolled, flipped or rotated
2. Another tile exists with the exact same pixels reordered
3. Another tile exists with identical intensitis shuffled across channels

We discuss each possibility below
1. The chance of an image being rolled is extremely low for a tiled image as it implies a redundancy that is likely unideal for memory storage. Instead one would expect an image to be formed through flips and rotations exactly serving our purpose.
2. More likely than 1 it is possible that a colour palette exists with another tile. However this is unlikely as such a combination would be redundant in colour space and not help build the richest image possible we assume the pixel artist would've wanted.
3. This is a valid wild card but seems highly unlikely. 

# Image Hashing for Near Perfect Tile Sets
NOTE: to be filled in later
