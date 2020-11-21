# The set of functions for imposing a total ordering on the simplices of the weight-filtered complex, constructing
# the boundary matrix and reducing the boundary matrix.

# Function to compare two simplices that helps create the total ordering of the simplices.
# Each simplex is represented as a list, bundled with its filter value: [simplex, filter value] e.g. [{0,1}, 4]
def compare(simplex1, simplex2):
    if len(simplex1[0]) == len(simplex2[0]):  # If both simplices have the same dimension
        if simplex1[1] == simplex2[1]:  # If both simplices have the same filter value
            if sum(simplex1[0]) > sum(simplex2[0]):  # Break the tie
                return 1