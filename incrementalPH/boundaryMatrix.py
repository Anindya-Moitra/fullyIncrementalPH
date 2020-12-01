# A set of functions for constructing the boundary matrix.

# Return the n-simplices and weights in a complex
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