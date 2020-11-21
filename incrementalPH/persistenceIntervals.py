# The set of functions for imposing a total ordering on the simplices of the weight-filtered complex, constructing
# the boundary matrix and reducing the boundary matrix.

# Function to compare two simplices that helps create the total ordering of the simplices.
# Each simplex is represented as a list, bundled with its filter value: [simplex, filter value] e.g. [{0,1}, 4]
def compare(item1, item2):
    if len(item1[0]) == len(item2[0]):
        if item1[1] == item2[1]:  # if both items have same filter value
            if sum(item1[0]) > sum(item2[0]):
                return 1