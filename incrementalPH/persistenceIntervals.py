# A set of functions to read the persistence intervals off the reduced matrix.

import matrixReduction as mr

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
        low_j = mr.low(j, reducedMatrices[0])
        if low_j == -1:
            intervalStart = [j, -1]
            intervals.append(intervalStart)  # -1 is a temporary placeholder until we update with death time

            # If no death time, then -1 represents a feature that does not die.

        else:  # death of feature
            feature = intervals.index([low_j, -1])  # Find the feature [start, end] so we can update the end point.
            intervals[feature][1] = j  # j is the death point

            # If the interval start point and end point are the same, then this feature begins and dies instantly.
            # So it is a useless interval, we don't want to waste memory keeping it.

            epsilonStart = filterValues[intervals[feature][0]]
            epsilonEnd = filterValues[j]
            if epsilonStart == epsilonEnd: intervals.remove(intervals[feature])

    return intervals