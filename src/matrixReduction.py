# Implementation of the standard algorithm for the reduction of the boundary matrix.
# Courtesy: A. Zomorodian and G. Carlsson, “Computing persistent homology,” 2005, and outlace.com

import numpy as np
from bisect import bisect
import boundaryMatrix as bm

# A function to return the row index of the lowest 1 (pivot) in a column i in the boundary matrix
def low(i, matrix):
    col = matrix[:, i]
    colLen = len(col)
    for i in range((colLen-1), -1, -1):  # Scan the column from the bottom until we find the first 1
        if col[i] == 1:
            return i

    # If the pivot of column i does not exist (e.g. in a column of all zeros), return -1 (pivot 'undefined')
    return -1


# A function to check if the boundary matrix is fully reduced
def isReduced(matrix):
    for j in range(matrix.shape[1]):  # Iterate through columns
        for i in range(j):  # Scan the columns until we reach column j
            low_j = low(j, matrix)
            low_i = low(i, matrix)
            if low_j == low_i and low_j != -1:
                return i, j  # Return column i that will be added to column j

    return [0, 0]


# Implementation of the standard algorithm to iteratively reduce the boundary matrix
def reduceBoundaryMatrix(matrix):
    reducedMatrix = matrix.copy()
    matrixShape = reducedMatrix.shape

    # 'memoryMatrix' will store the history of column additions
    memoryMatrix = np.identity(matrixShape[1], dtype=np.uint8)
    r = isReduced(reducedMatrix)

    while (r != [0, 0]):
        i = r[0]
        j = r[1]
        col_j = reducedMatrix[:, j]
        col_i = reducedMatrix[:, i]
        reducedMatrix[:, j] = np.bitwise_xor(col_i, col_j)  # Add column i to j
        memoryMatrix[i, j] = 1
        r = isReduced(reducedMatrix)

    return reducedMatrix, memoryMatrix

def delColsRows(reducedMatrix, memoryMatrix, delIndices):

    # A function to incrementally delete the columns and rows corresponding to the deleted simplices
    # from the reduced boundary matrix and memory matrix.

    return reducedMatrix, memoryMatrix


def addColsRows(reducedMatrix, memoryMatrix, newSimplices, sortedSimplices):

    # A function to incrementally add the columns and rows corresponding to the already
    # reduced boundary matrix, and then further reduce the matrix, if needed.

    for simplex, weight in zip(newSimplices[0], newSimplices[1]):
        # For each simplex in the list of new simplices that needs to be added to the reduced
        # boundary matrix, first find its index with respect to the total ordering.
        allSimplices = sortedSimplices[0]  # The first (0-th) element is the list of simplices.

        col_j = []
        rowIndex = 0

        # Construct the column for the new simplex
        for existingSimplex in allSimplices:

            # Apply the 'compression' technique while constructing the column
            entry = bm.checkFace(existingSimplex, simplex)

            if entry == 1:
                # Check if the 'rowIndex' column of the reduced matrix is all zeros
                col_i = reducedMatrix[:, rowIndex]
                if 1 in col_i:  # If the 'rowIndex' column of the reduced matrix is not zero:
                    entry = 0

            col_j.append(entry)
            col_j.append(0)  # The last additional entry
            rowIndex += 1

        # Insert a row and column to the reduced matrix for the current simplex.
        dim_reducedMatrix = reducedMatrix.shape[0]  # It is a sqaure matrix

        # The last row of the matrix is always zero because the highest dimensional simplex
        # cannot be a face of another simplex in the filtration.
        lastRow = np.zeros(dim_reducedMatrix)

        reducedMatrix = np.append(reducedMatrix, lastRow,
                                  axis=0)  # Add the zero row to the bottom of the matrix


        # Find the index/position of the current simplex within the reduced boundary matrix

        # sortedSimplices[1] is the sorted list of weights of the existing simplices in the filtration
        insersionIndex = bisect(sortedSimplices[1], weight)
        reducedMatrix = np.hstack((reducedMatrix[:, :insersionIndex], col_j, reducedMatrix[:, insersionIndex:]))

        # Update the existing weight-filtered complex
        sortedSimplices[0].insert(insersionIndex, simplex)
        sortedSimplices[1].insert(insersionIndex, weight)


    return reducedMatrix, memoryMatrix, sortedSimplices
