# A set of functions to read the persistence intervals off the reduced matrix.

import matrixRe

# 'reducedMatrices' is a tuple of two matrices: the reduced boundary matrix and the memory matrix
def readIntervals(reducedMatrices, filterValues):

    # Store persistence intervals as a list of 2-element lists, e.g. [2, 4]: start at "time" point 2, end at
    # "time" point 4. For now, the "time" points are just the indices from the boundary matrix, that will be
    # converted to epsilon values later.
    intervals = []

    # Loop through each column j:
    # If low(j) = -1 (undefined, all zeros) then j represents the birth of a new feature j
    # if low(j) = i (defined), then j represents the death of feature i
    for j in range(reducedMatrices[0].shape[1]):  # For each column
        low_j = low(j, reducedMatrices[0])
        if low_j == -1:
            intervalStart = [j, -1]
            intervals.append(intervalStart)  # -1 is a temporary placeholder until we update with death time
            