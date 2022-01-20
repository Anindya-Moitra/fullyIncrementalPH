# Determine whether the current vector belongs to any of the existing partitions.
def determineMembership(nnDistsFrmCurrVecToPartns, avgNNDistPartitions, f2, f3):
    # Find the minimum of the distances from the current vector to the existing partitions.
    minDistFrmCurrVecToPartn = min(nnDistsFrmCurrVecToPartns.values())

    # Find the partition(s) that is (are) nearest to the current vector.
    nearestPartitions = [k for k in nnDistsFrmCurrVecToPartns
                         if nnDistsFrmCurrVecToPartns[k] == minDistFrmCurrVecToPartn]

    # If the (nearest neighbor) distance from the current vector to any partition is 0, assign the current
    # vector to that partition. If there are more than one such partitions, assign the current vector randomly
    # to one of them.
    if minDistFrmCurrVecToPartn == 0:
        assignedPartition = random.choice(nearestPartitions)
        return assignedPartition

    # Find the max of the avg. nearest neighbor distances of the existing partitions in the window.
    # maxAvgNNdPartns = max(avgNNDistPartitions.values())

    existingLabels = list(avgNNDistPartitions.keys())  # Create a list of existing partition labels in the window.
    maxLabel = max(existingLabels)  # Find the max. of the existing partition labels.

    # If the minimum distance from the current vector to existing partitions is higher than f2,
    # assign a new partition to the current vector.
    if minDistFrmCurrVecToPartn > f2:
        return maxLabel + 1

    candidatePartns = []
    for partn in avgNNDistPartitions:
        if avgNNDistPartitions[partn] == -1 and nnDistsFrmCurrVecToPartns[partn] <= f2:
            candidatePartns.append(partn)

        elif avgNNDistPartitions[partn] <= f3 and nnDistsFrmCurrVecToPartns[partn] <= 1:
            candidatePartns.append(partn)

        elif avgNNDistPartitions[partn] > 0 and nnDistsFrmCurrVecToPartns[partn] / avgNNDistPartitions[partn] <= f2:
            candidatePartns.append(partn)

    nearestCandidatePartns = list(set(nearestPartitions) & set(candidatePartns))

    if len(nearestCandidatePartns) != 0:
        assignedPartition = max(nearestCandidatePartns)
        return assignedPartition

    return maxLabel + 1
