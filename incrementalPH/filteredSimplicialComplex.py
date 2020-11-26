# A set of functions for building a filtered Vietoris-Rips complex from data point(s)
# Courtesy: A. Zomorodian, "Fast construction of the Vietoris-Rips complex", 2010, and outlace.com

import numpy as np
import itertools


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


def lowerNbrs(vertexSet, edgeSet, vertex):  # Lowest neighbors based on the ordering of simplices
    return {x for x in vertexSet if {x, vertex} in edgeSet and vertex > x}

def incrementalRipsComplex(vertices, edges, weights, k):  # k is the maximal dimension we want to compute
    ripsComplex = [{n} for n in vertices]
    filterValues = [0 for j in ripsComplex]  # A vertex has a weight of 0
    for i in range(len(edges)):  # Add 1-simplices (edges) and associated filter values
        ripsComplex.append(edges[i])
        filterValues.append(weights[i])

    if k > 1:
        for i in range(k):
            for simplex in [x for x in ripsComplex if len(x) == i+2]:  # Skip 0-simplices and 1-simplices
                # For each uertex in the simplex
                nbrs = set.intersection(*[lowerNbrs(vertices, edges, z) for z in simplex])
                for nbr in nbrs:
                    newSimplex = set.union(simplex, {nbr})
                    ripsComplex.append(newSimplex)
                    filterValues.append(getFilterValue(newSimplex, ripsComplex, filterValues))

    return sortComplex(ripsComplex, filterValues)  # Sort simplices according to filter values


def getFilterValue(simplex, edges, weights):  # Filter value is the maximum weight of an edge in the simplex
    oneSimplices = list(itertools.combinations(simplex, 2))  # Get set of 1-simplices in the simplex
    maxWeight = 0
    for oneSimplex in oneSimplices:
        filterValue = weights[edges.index(set(oneSimplex))]
        if filterValue > maxWeight: maxWeight = filterValue
    return maxWeight