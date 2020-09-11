import numpy as np
from scipy.spatial import distance_matrix
import statistics
import random


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

        if window.shape[0] == windowMaxSize:   # Once the window size reaches its max:
            distMat = np.tril(distance_matrix(window, window))   # Construct the lower triangular distance matrix.
