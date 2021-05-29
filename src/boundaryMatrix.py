# A set of functions for constructing the boundary matrix.
# Courtesy: outlace.com

import numpy as np

# A function to return the n-simplices and weights of a complex
def nSimplices(n, filteredComplex):
    nChain = []
    nFilters = []
    for i in range(len(filteredComplex[0])):
        simplex = filteredComplex[0][i]
        if len(simplex) == (n+1):
            nChain.append(simplex)
            nFilters.append(filteredComplex[1][i])
    if (nChain == []): nChain = [0]
    return nChain, nFilters


# A function to check if simplex is a face of another simplex
def checkFace(face, simplex):
    if simplex == 0:
        return 1
    elif (set(face) < set(simplex) and (len(face) == (len(simplex) - 1))):  # If face is a (n-1) subset of simplex
        return 1
    else:
        return 0


# A function to construct the boundary matrix
def buildBoundaryMatrix(filteredComplex):
    boundaryMatrix = np.zeros((len(filteredComplex[0]), len(filteredComplex[0])), dtype=np.uint8)

    i = 0
    for colSimplex in filteredComplex[0]:
        j = 0
        for rowSimplex in filteredComplex[0]:
            boundaryMatrix[j, i] = checkFace(rowSimplex, colSimplex)
            j += 1
        i += 1
    return boundaryMatrix