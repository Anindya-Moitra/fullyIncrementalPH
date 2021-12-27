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

            # Extract the average nearest neighbor distance in the single 'partition' in the window.
            avgNNDistSinglePartition = list(avgNNDistPartitions.values())[0]

            if nnDistCurrVec == 0:
                continue

            if avgNNDistSinglePartition <= f3 and nnDistCurrVec <= 1:
                continue

            if avgNNDistSinglePartition == 0 or nnDistCurrVec / avgNNDistSinglePartition > f2:
                deletedKey = windowKeys.pop(0)   # Delete the key (the lowest key) from the front of the list.
                deletedLabel = partitionLabels.pop(0)  # Delete the label from the front of the list.

                deletedVector = window[0]  # Verify
                window = np.delete(window, 0, axis=0)  # Delete the vector from the front of the sliding window.

                # Get the vertex, edges and weights that are being deleted from the neighborhood graph and
                # from the complex. Reuse the 'buildGraph' function.
                vertexDel, edgesDel, weightsDel = fsc.buildGraph(dataPoints=deletedVector,
                                                                 epsilon=eps, metric=euclidianDist)

                # Delete the simplices corresponding to 'vertexDel' from the filtration
                sortedSimplices, delIndices = fsc.deleteSimplices(sortedSimplices, vertexDel)

                # Delete the columns and rows corresponding to the deleted simplices from the reduced
                # boundary matrix and memory matrix.
                reducedMatrix, memoryMatrix = mr.delColsRows(reducedMatrix, memoryMatrix, delIndices)

                # Delete the 0-th row and 0-th column from the distance matrix.
                distMat = np.delete(np.delete(distMat, 0, axis=0), 0, axis=1)

                # Delete the corresponding distance value from the list of distances from the current vector
                # to the existing ones in the window.
                distsFromCurrVec.pop(0)

                # Recompute the average nearest neighbor distance in the existing partition.
                nnDistsPartition = []
                nnDist0thPoint = min(distMat[1:, 0])
                nnDistsPartition.append(nnDist0thPoint)

                for index in range(1, windowMaxSize-1):
                    row = distMat[index, :index]
                    column = distMat[index + 1:, index]
                    distsFromPoint = np.append(row, column)
                    nnDistPoint = min(distsFromPoint)
                    nnDistsPartition.append(nnDistPoint)

                avgNNDistPartition = statistics.mean(nnDistsPartition)
                avgNNDistPartitions[deletedLabel] = avgNNDistPartition

                # Decrement the number of points in the existing partition by 1.
                numPointsPartn[deletedLabel] = windowMaxSize - 1

                # Insert the current vector, its key and a new label into the rear ends
                # of the corresponding containers.
                label = deletedLabel + 1
                window = np.append(window, currVec, axis=0)
                windowKeys.append(key)
                partitionLabels.append(label)

                maxKeys[label] = key
                key += 1

                InsertedVector = currVec

                # Get the vertex, edges and weights that are being added to the neighborhood graph and
                # to the complex.
                vertexAdd, edgesAdd, weightsAdd = fsc.buildGraph(dataPoints=InsertedVector,
                                                                 epsilon=eps, metric=euclidianDist)

                sortedSimplices = fsc.incrementalRipsComplex(vertexAdd, edgesAdd, weightsAdd, k)

                reducedMatrix, memoryMatrix = mr.addColsRows(reducedMatrix, memoryMatrix, newSimplices)

                # Update the distance matrix.
                distsFromCurrVecArray = np.array(distsFromCurrVec).reshape(1, windowMaxSize - 1)
                distMat = np.append(distMat, distsFromCurrVecArray, axis=0)  # Add a row to the bottom of the matrix.
                zeroColumn = np.array([0] * windowMaxSize).reshape(windowMaxSize, 1)
                distMat = np.append(distMat, zeroColumn, axis=1)  # Add a column to the right of the matrix.

                # Add a new key, value pair to the dictionary of partitions and their average nearest neighbor
                # distances. In this case, however, the newly created partition has only one point. So, at this
                # time, we insert a value of -1 for the average nearest neighbor distance of the new point.
                avgNNDistPartitions[label] = -1

                numPointsPartn[label] = 1


        else:
            # Create a dictionary to store the nearest neighbor distance from the current vector to each
            # partition in the window.
            nnDistsFrmCurrVecToPartns = {}

            # Create a list to store the distances from the current vector to all existing points in the window.
            distsFromCurrVec = [0] * windowMaxSize

            for partition in avgNNDistPartitions:
                # Find the positions of the points (in the window) that are members of the present 'partition'.
                indicesOfMembers = [i for i, pl in enumerate(partitionLabels) if pl == partition]

                # Compute the distances from the current vector to the members of the current partition.
                distsFrmCurrVecToMembrs = []
                for idx in indicesOfMembers:
                    member = window[idx]
                    member.shape = (1, dim)
                    dist = np.linalg.norm(member - currVec)
                    distsFrmCurrVecToMembrs.append(dist)
                    distsFromCurrVec[idx] = dist

                # Sort the distances from the current vector to the members in the present partition
                # in increasing order.
                ascendingDistsToMembrs = sorted(distsFrmCurrVecToMembrs)

                # Find the distance from the current vector to its nearest neighbor in the present partition.
                nndToPartn = ascendingDistsToMembrs[0]

                # Insert the distance from the current vector to its nearest neighbor in the present partition
                # into the corresponding dictionary.
                nnDistsFrmCurrVecToPartns[partition] = nndToPartn

            # Determine the membership of the current vector to one of the existing partitions in the window.
            # If the current vector cannot be assigned to any of the existing partitions, create a new partition
            # with only the current vector.
            targetPartition = determineMembership(nnDistsFrmCurrVecToPartns, avgNNDistPartitions, f2, f3)

            # Find the outdated partition(s).
            outdatedPartn = [op for op in maxKeys if (key - maxKeys[op]) > timeToBeOutdated]

            if len(outdatedPartn) != 0:  # If there is (are) outdated partition(s):
                # Find the number(s) of points in the outdated partition(s).
                numPtsOutdated = [numPointsPartn[npo] for npo in outdatedPartn]

                # Find the smallest size (i.e. the min. of the number(s) of points) of the outdated partition(s).
                numPtsSmallestOutdated = min(numPtsOutdated)

                # Find the smallest outdated partition(s).
                smallestOutdated = [so for so, numPts in numPointsPartn.items() if numPts == numPtsSmallestOutdated]

                partnToBeDeleted = min(smallestOutdated)

                # Find the first occurrence of a partition label from which deletion will take place.
                for i in range(windowMaxSize):
                    if partitionLabels[i] == partnToBeDeleted:
                        deletedLabel = partnToBeDeleted
                        indexToBeDeleted = i
                        del partitionLabels[i]  # Delete the partition label.
                        break

                del windowKeys[indexToBeDeleted]  # Delete the key of the vector.

                deletedVector = window[indexToBeDeleted]

                # Get the vertex, edges and weights that are being deleted from the neighborhood graph and
                # from the complex. Reuse the 'buildGraph' function.
                vertexDel, edgesDel, weightsDel = fsc.buildGraph(dataPoints=deletedVector,
                                                                 epsilon=eps, metric=euclidianDist)

                # Delete the simplices corresponding to 'vertexDel' from the filtration
                sortedSimplices, delIndices = fsc.deleteSimplices(sortedSimplices, vertexDel)

                # Delete the columns and rows corresponding to the deleted simplices from the reduced
                # boundary matrix and memory matrix.
                reducedMatrix, memoryMatrix = mr.delColsRows(reducedMatrix, memoryMatrix, delIndices)

                window = np.delete(window, indexToBeDeleted, axis=0)  # Delete the vector from the sliding window.

                # Delete the corresponding row and column from the distance matrix.
                distMat = np.delete(np.delete(distMat, indexToBeDeleted, axis=0), indexToBeDeleted, axis=1)

                # Delete the corresponding distance value from the list of distances from the current vector
                # to the existing ones in the window.
                del distsFromCurrVec[indexToBeDeleted]

                # Find the positions of the points (in the window) that are members of the partition
                # from which the point was deleted.
                delPmemIndices = [i for i, pl in enumerate(partitionLabels) if pl == deletedLabel]

                # If there are no more points left in the partition from which the deletion took place:
                if not delPmemIndices:
                    del avgNNDistPartitions[deletedLabel]
                    del numPointsPartn[deletedLabel]
                    del maxKeys[deletedLabel]

                else:
                    # Decrement the number of points in the partition from which the point was deleted by 1.
                    numPointsPartn[deletedLabel] = numPtsSmallestOutdated - 1

                    # Recompute the average nearest neighbor distance in the partition from which the
                    # point was deleted.
                    # If there is only 1 point left in the partition that the point was deleted from:
                    if numPointsPartn[deletedLabel] == 1:
                        avgNNDistPartitions[deletedLabel] = -1

                    else:
                        nnDistsDelPartition = []
                        for i in delPmemIndices:
                            row = [distMat[i, :i][j] for j in delPmemIndices if j < i]
                            column = [distMat[i + 1:, i][j - (i + 1)] for j in delPmemIndices if j > i]
                            distsFromPoint = np.append(row, column)
                            nnDistPoint = min(distsFromPoint)
                            nnDistsDelPartition.append(nnDistPoint)

                        avgNNdDelPartition = statistics.mean(nnDistsDelPartition)
                        avgNNDistPartitions[deletedLabel] = avgNNdDelPartition

                # Insert the current vector, its key and partition label into the rear ends
                # of the corresponding containers.
                window = np.append(window, currVec, axis=0)
                windowKeys.append(key)
                partitionLabels.append(targetPartition)
                maxKeys[targetPartition] = key  # Insert or update the maxKey of the target partition.

                # Update the distance matrix.
                distsFromCurrVecArray = np.array(distsFromCurrVec).reshape(1, windowMaxSize - 1)
                distMat = np.append(distMat, distsFromCurrVecArray,
                                    axis=0)  # Add a row to the bottom of the matrix.

                zeroColumn = np.array([0] * windowMaxSize).reshape(windowMaxSize, 1)
                distMat = np.append(distMat, zeroColumn, axis=1)  # Add a column to the right of the matrix.

                InsertedVector = currVec

                # Get the vertex, edges and weights that are being added to the neighborhood graph and
                # to the complex.
                vertexAdd, edgesAdd, weightsAdd = fsc.buildGraph(dataPoints=InsertedVector,
                                                                 epsilon=eps, metric=euclidianDist)

                sortedSimplices = fsc.incrementalRipsComplex(vertexAdd, edgesAdd, weightsAdd, k)

                reducedMatrix, memoryMatrix = mr.addColsRows(reducedMatrix, memoryMatrix, newSimplices)

                if targetPartition not in avgNNDistPartitions:  # If the current vector was assigned a new partition:
                    avgNNDistPartitions[targetPartition] = -1
                    numPointsPartn[targetPartition] = 1

                else:  # The current vector is assigned to one of the existing partitions:
                    # Retrieve the avg. nearest neighbor distance in the target partition.
                    avgNNdTP = avgNNDistPartitions[targetPartition]

                    if avgNNdTP == -1:   # If the target partition previously had only 1 point:
                        # Update the avg. nearest neighbor distance of the partition the current vector was added to.
                        avgNNDistPartitions[targetPartition] = nnDistsFrmCurrVecToPartns[targetPartition]
                        numPointsPartn[targetPartition] = 2

                    else:
                        # Find the positions of the points (in the window) that are members of the target partition.
                        tpMemIndices = [i for i, tp in enumerate(partitionLabels) if tp == targetPartition]

                        # Recompute the average nearest neighbor distance in the target partition.
                        nnDistsTP = []
                        for i in tpMemIndices:
                            row = [distMat[i, :i][j] for j in tpMemIndices if j < i]
                            column = [distMat[i+1:, i][j - (i+1)] for j in tpMemIndices if j > i]
                            distsFromPoint = np.append(row, column)
                            nnDistPoint = min(distsFromPoint)
                            nnDistsTP.append(nnDistPoint)

                        avgNNdTargetPartn = statistics.mean(nnDistsTP)
                        avgNNDistPartitions[targetPartition] = avgNNdTargetPartn
                        numPointsPartn[targetPartition] = numPointsPartn[targetPartition] + 1

                key += 1

            else:  # There is no outdated partition in the window:
                if targetPartition not in avgNNDistPartitions:  # If the current vector was assigned a new partition:
                    deletedKey = windowKeys.pop(0)  # Delete the key (the lowest key) from the front of the list.
                    deletedLabel = partitionLabels.pop(0)  # Delete the label from the front of the list.
                    window = np.delete(window, 0, axis=0)  # Delete the vector from the front of the sliding window.

                    deletedVector = window[0]

                    # Get the vertex, edges and weights that are being deleted from the neighborhood graph and
                    # from the complex. Reuse the 'buildGraph' function.
                    vertexDel, edgesDel, weightsDel = fsc.buildGraph(dataPoints=deletedVector,
                                                                     epsilon=eps, metric=euclidianDist)

                    # Delete the simplices corresponding to 'vertexDel' from the filtration
                    sortedSimplices, delIndices = fsc.deleteSimplices(sortedSimplices, vertexDel)

                    # Delete the columns and rows corresponding to the deleted simplices from the reduced
                    # boundary matrix and memory matrix.
                    reducedMatrix, memoryMatrix = mr.delColsRows(reducedMatrix, memoryMatrix, delIndices)

                    # Delete the 0-th row and 0-th column from the distance matrix.
                    distMat = np.delete(np.delete(distMat, 0, axis=0), 0, axis=1)

                    # Delete the corresponding distance value from the list of distances from the current vector
                    # to the existing ones in the window.
                    distsFromCurrVec.pop(0)

                    # Find the positions of the points (in the window) that are members of the partition
                    # from which the point was deleted.
                    delPmemIndices = [i for i, pl in enumerate(partitionLabels) if pl == deletedLabel]

                    # If there are no more points left in the partition from which the deletion took place:
                    if not delPmemIndices:
                        del avgNNDistPartitions[deletedLabel]
                        del numPointsPartn[deletedLabel]
                        del maxKeys[deletedLabel]

                    else:
                        # Decrement the number of points in the partition from which the point was deleted by 1.
                        numPointsPartn[deletedLabel] = numPointsPartn[deletedLabel] - 1

                        # Recompute the average nearest neighbor distance in the partition from which the
                        # point was deleted.

                        # If there is only 1 point left in the partition that the point was deleted from:
                        if numPointsPartn[deletedLabel] == 1:
                            avgNNDistPartitions[deletedLabel] = -1

                        else:
                            nnDistsDelPartition = []
                            for i in delPmemIndices:
                                row = [distMat[i, :i][j] for j in delPmemIndices if j < i]
                                column = [distMat[i + 1:, i][j - (i + 1)] for j in delPmemIndices if j > i]
                                distsFromPoint = np.append(row, column)
                                nnDistPoint = min(distsFromPoint)
                                nnDistsDelPartition.append(nnDistPoint)

                            avgNNdDelPartition = statistics.mean(nnDistsDelPartition)
                            avgNNDistPartitions[deletedLabel] = avgNNdDelPartition

                    # Insert the current vector, its key and the new label into the rear ends
                    # of the corresponding containers.
                    window = np.append(window, currVec, axis=0)
                    windowKeys.append(key)
                    partitionLabels.append(targetPartition)
                    maxKeys[targetPartition] = key

                    key += 1

                    # Update the distance matrix.
                    distsFromCurrVecArray = np.array(distsFromCurrVec).reshape(1, windowMaxSize - 1)
                    distMat = np.append(distMat, distsFromCurrVecArray, axis=0)
                    zeroColumn = np.array([0] * windowMaxSize).reshape(windowMaxSize, 1)
                    distMat = np.append(distMat, zeroColumn, axis=1)  # Add a column to the right of the matrix.

                    InsertedVector = currVec

                    # Get the vertex, edges and weights that are being added to the neighborhood graph and
                    # to the complex.
                    vertexAdd, edgesAdd, weightsAdd = fsc.buildGraph(dataPoints=InsertedVector,
                                                                     epsilon=eps, metric=euclidianDist)

                    sortedSimplices = fsc.incrementalRipsComplex(vertexAdd, edgesAdd, weightsAdd, k)

                    reducedMatrix, memoryMatrix = mr.addColsRows(reducedMatrix, memoryMatrix, newSimplices)

                    # Add a new key, value pair to the dictionary of partitions and their average nearest neighbor
                    # distances. In this case, however, the newly created partition has only one point. So, at this
                    # time, we insert a value of -1 for the average nearest neighbor distance of the new point.
                    avgNNDistPartitions[targetPartition] = -1
                    numPointsPartn[targetPartition] = 1

                else:  # The current vector is assigned to one of the existing partitions:
                    # Find the first occurrence of a partition label != targetPartition label.
                    for i in range(windowMaxSize):
                        if partitionLabels[i] != targetPartition:
                            deletedLabel = partitionLabels[i]
                            indexToBeDeleted = i
                            del partitionLabels[i]  # Delete the partition label.
                            break

                    del windowKeys[indexToBeDeleted]  # Delete the key of the vector.

                    deletedVector = window[indexToBeDeleted]

                    # Get the vertex, edges and weights that are being deleted from the neighborhood graph and
                    # from the complex. Reuse the 'buildGraph' function.
                    vertexDel, edgesDel, weightsDel = fsc.buildGraph(dataPoints=deletedVector,
                                                                     epsilon=eps, metric=euclidianDist)

                    # Delete the simplices corresponding to 'vertexDel' from the filtration
                    sortedSimplices, delIndices = fsc.deleteSimplices(sortedSimplices, vertexDel)

                    # Delete the columns and rows corresponding to the deleted simplices from the reduced
                    # boundary matrix and memory matrix.
                    reducedMatrix, memoryMatrix = mr.delColsRows(reducedMatrix, memoryMatrix, delIndices)

                    window = np.delete(window, indexToBeDeleted, axis=0)  # Delete the vector from the sliding window.

                    # Delete the corresponding row and column from the distance matrix.
                    distMat = np.delete(np.delete(distMat, indexToBeDeleted, axis=0), indexToBeDeleted, axis=1)

                    # Delete the corresponding distance value from the list of distances from the current vector
                    # to the existing ones in the window.
                    del distsFromCurrVec[indexToBeDeleted]

                    # Find the positions of the points (in the window) that are members of the partition
                    # from which the point was deleted.
                    delPmemIndices = [i for i, pl in enumerate(partitionLabels) if pl == deletedLabel]

                    # If there are no more points left in the partition from which the deletion took place:
                    if not delPmemIndices:
                        del avgNNDistPartitions[deletedLabel]
                        del numPointsPartn[deletedLabel]
                        del maxKeys[deletedLabel]

                    else:
                        # Decrement the number of points in the partition from which the point was deleted by 1.
                        numPointsPartn[deletedLabel] = numPointsPartn[deletedLabel] - 1

                        # Recompute the average nearest neighbor distance in the partition from which the
                        # point was deleted.
                        # If there is only 1 point left in the partition that the point was deleted from:
                        if numPointsPartn[deletedLabel] == 1:
                            avgNNDistPartitions[deletedLabel] = -1

                        else:
                            nnDistsDelPartition = []
                            for i in delPmemIndices:
                                row = [distMat[i, :i][j] for j in delPmemIndices if j < i]
                                column = [distMat[i + 1:, i][j - (i + 1)] for j in delPmemIndices if j > i]
                                distsFromPoint = np.append(row, column)
                                nnDistPoint = min(distsFromPoint)
                                nnDistsDelPartition.append(nnDistPoint)

                            avgNNdDelPartition = statistics.mean(nnDistsDelPartition)
                            avgNNDistPartitions[deletedLabel] = avgNNdDelPartition

                    # Insert the current vector, its key and partition label into the rear ends
                    # of the corresponding containers.
                    window = np.append(window, currVec, axis=0)
                    windowKeys.append(key)
                    partitionLabels.append(targetPartition)
                    maxKeys[targetPartition] = key  # Update the maxKey of the target partition.

                    key += 1

                    # Update the distance matrix.
                    distsFromCurrVecArray = np.array(distsFromCurrVec).reshape(1, windowMaxSize - 1)
                    distMat = np.append(distMat, distsFromCurrVecArray, axis=0)  # Add a row to the bottom of the matrix
                    zeroColumn = np.array([0] * windowMaxSize).reshape(windowMaxSize, 1)
                    distMat = np.append(distMat, zeroColumn, axis=1)  # Add a column to the right of the matrix.

                    InsertedVector = currVec

                    # Get the vertex, edges and weights that are being added to the neighborhood graph and
                    # to the complex.
                    vertexAdd, edgesAdd, weightsAdd = fsc.buildGraph(dataPoints=InsertedVector,
                                                                     epsilon=eps, metric=euclidianDist)

                    sortedSimplices = fsc.incrementalRipsComplex(vertexAdd, edgesAdd, weightsAdd, k)

                    reducedMatrix, memoryMatrix = mr.addColsRows(reducedMatrix, memoryMatrix, newSimplices)

                    # Retrieve the avg. nearest neighbor distance in the target partition.
                    avgNNdTP = avgNNDistPartitions[targetPartition]

                    # Update the distance matrix.
                    distsFromCurrVecArray = np.array(distsFromCurrVec).reshape(1, windowMaxSize - 1)
                    distMat = np.append(distMat, distsFromCurrVecArray, axis=0)  # Add a row to the bottom of the matrix
                    zeroColumn = np.array([0] * windowMaxSize).reshape(windowMaxSize, 1)
                    distMat = np.append(distMat, zeroColumn, axis=1)  # Add a column to the right of the matrix.

                    # Retrieve the avg. nearest neighbor distance in the target partition.
                    avgNNdTP = avgNNDistPartitions[targetPartition]

                    if avgNNdTP == -1:  # If the target partition previously had only 1 point:
                        # Update the avg. nearest neighbor distance of the partition the current vector was added to.
                        avgNNDistPartitions[targetPartition] = nnDistsFrmCurrVecToPartns[targetPartition]
                        numPointsPartn[targetPartition] = 2

                    else:
                        # Find the positions of the points (in the window) that are members of the target partition.
                        tpMemIndices = [i for i, tp in enumerate(partitionLabels) if tp == targetPartition]

                        # Recompute the average nearest neighbor distance in the target partition.
                        nnDistsTP = []
                        for i in tpMemIndices:
                            row = [distMat[i, :i][j] for j in tpMemIndices if j < i]
                            column = [distMat[i + 1:, i][j - (i + 1)] for j in tpMemIndices if j > i]
                            distsFromPoint = np.append(row, column)
                            nnDistPoint = min(distsFromPoint)
                            nnDistsTP.append(nnDistPoint)

                        avgNNdTargetPartn = statistics.mean(nnDistsTP)
                        avgNNDistPartitions[targetPartition] = avgNNdTargetPartn
                        numPointsPartn[targetPartition] = numPointsPartn[targetPartition] + 1
