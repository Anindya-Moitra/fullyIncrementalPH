import numpy as np
# from scipy.spatial import distance_matrix
import statistics
import random

import filteredSimplicialComplex as fsc
import boundaryMatrix as bm
import matrixReduction as mr


data = np.genfromtxt('testData.csv', delimiter=',', skip_header=1)   # Load the entire data as a numpy array.
dim = data.shape[1]   # Get the dimension of the data.

window = np.empty([0, dim])   # Create an empty sliding window.
windowMaxSize = 200   # Set the maximum number of data points the window may contain.
pointCounter = 0

windowKeys = []
partitionLabels = []
avgNNDistPartitions = {}

key = 0

# f1 = 0.5
f2 = 4
f3 = 0.25
eps = 5  # The scale parameter for the computation of persistent homology
k = 1  # The maximum dimension up to which persistent homology will be computed

# A partition is considered outdated if it did not receive any new point for more than
# the last 'timeToBeOutdated' insertions.
timeToBeOutdated = 10

numPointsPartn = {}   # Create a dictionary to store the number of points in each partition.
maxKeys = {}   # Create a dictionary to store the maxKey of each partition.

# Loop through each vector in the data: this is analogous to processing data objects from
# a stream, one at a time.
for currVec in data:
    pointCounter += 1
    currVec.shape = (1, dim)

    # Initialize the sliding window. During the initialization, let's assume all points from the stream
    # belong to Partition 0.  Besides, we will not compute/update the persistence intervals during the
    # initialization phase.
    if window.shape[0] < windowMaxSize:
        label = 0
        window = np.append(window, currVec, axis=0)   # Fill the window until it reaches its max size.
        windowKeys.append(key)
        partitionLabels.append(label)
        key += 1

        # Once the window size reaches its max size, construct the simplicial complex and compute the
        # persistence intervals for the first time.
        if window.shape[0] == windowMaxSize:

            # Construct the neighborhood graph based on the scale parameter epsilon
            vertices, edges, weights = fsc.buildGraph(dataPoints=window,
                                                      epsilon=eps, metric=euclidianDist)

            # Expand the neighborhood graph into a weight-filtered Vietoris-Rips filtration
            sortedSimplices = fsc.incrementalRipsComplex(vertices, edges, weights, k)

            # Construct the boundary matrix and reduce it
            boundaryMatrix = bm.buildBoundaryMatrix(sortedSimplices)
            reducedMatrix, memoryMatrix = mr.reduceBoundaryMatrix(boundaryMatrix)

            # Find the average nearest neighbor distance in the existing partition (i.e. Partition 0).
            nnDistsPartition0 = []
            nnDist0thPoint = min(distMat[1:, 0])
            nnDistsPartition0.append(nnDist0thPoint)
            for index in range(1, windowMaxSize):
                row = distMat[index, :index]
                column = distMat[index + 1:, index]
                distsFromPoint = np.append(row, column)
                nnDistPoint = min(distsFromPoint)
                nnDistsPartition0.append(nnDistPoint)

            avgNNDistPartition0 = statistics.mean(nnDistsPartition0)
            avgNNDistPartitions[label] = avgNNDistPartition0

            numPointsPartn[label] = windowMaxSize
            maxKeys[label] = key - 1

    else:
        if len(avgNNDistPartitions) == 1:   # If the window is 'pure':
            # Compute the distances from the current vector to the existing ones in the window.
            distsFromCurrVec = []
            for existingVector in window:
                existingVector.shape = (1, dim)
                dist = np.linalg.norm(existingVector - currVec)
                distsFromCurrVec.append(dist)

            # Sort the distances from the current vector (to the existing ones in the window) in increasing order.
            ascendingDists = sorted(distsFromCurrVec)

            # Find the distance from the current vector to its nearest neighbor in the window.
            nnDistCurrVec = ascendingDists[0]

            print("nnDistCurrVec")
            print(nnDistCurrVec)

            # Extract the average nearest neighbor distance in the single 'partition' in the window.
            avgNNDistSinglePartition = list(avgNNDistPartitions.values())[0]
