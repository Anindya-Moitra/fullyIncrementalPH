# A set of functions for building a filtered Vietoris-Rips complex from data point(s)
# Courtesy: A. Zomorodian, "Fast construction of the Vietoris-Rips complex", 2010
# outlace.com

import numpy as np
import matplotlib.pyplot as plt


def euclidianDist(a, b):
  return np.linalg.norm(a - b)  # Euclidian distance metric


# Build neighorhood graph
def buildGraph(dataPoints, epsilon, metric=euclidianDist):  # dataPoints is a numpy array
  vertices = [x for x in range(dataPoints.shape[0])]  # Initialize vertex set, reference indices from original data array
  edges = []  # Initialize empty edge array
  weights = []  # Initialize weight array, store the weight (which in this case is the distance) for each edge
  for i in range(dataPoints.shape[0]):  # Iterate through each data point
      for j in range(dataPoints.shape[0]-i):  # Inner loop to calculate pairwise distances between data points
          a = dataPoints[i]
          b = dataPoints[j+i]  # Each simplex is a set, hence [0, 1] = [1, 0]; so only store one
          if (i != j+i):
              dist = metric(a, b)
              if dist <= epsilon:
                  edges.append({i, j+i})  # Add edge if distance between points is less than epsilon
                  weights.append([len(edges)-1, dist])  # Store index and weight
  return vertices, edges, weights