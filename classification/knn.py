"""
k nearest neighbours classifier
"""


import csv
import numpy as np
from random import choice
from collections import defaultdict
from operator import itemgetter


def load_data(input_indexes, output_index, filename, has_header=True):
    inputs = []
    outputs = []

    input_getter = itemgetter(*input_indexes)
    output_getter = itemgetter(output_index)
    with open(filename) as fd:
        reader = csv.reader(fd)
        for i, row in enumerate(reader):
            # Ignore first row if our csv has header
            if has_header and i == 0:
                continue

            input_row = input_getter(row)
            # Always make input row a tuple, now matter how many
            # items are there.
            if len(input_indexes) == 1:
                input_row = (input_row, )
            inputs.append([float(x) for x in input_row])
            outputs.append(output_getter(row))

    return np.array(inputs), outputs


def sample_indices(data, partition_size):
    available_indices = range(len(data))
    sample_indices = []

    while len(sample_indices) != partition_size:
        sample_idx = choice(available_indices)
        sample_indices.append(sample_idx)
        available_indices.pop(available_indices.index(sample_idx))

    return sample_indices, available_indices


def classify(X, data, classes, k=3):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    distances = []
    for i, item in enumerate(data):
        diff = X - item
        diff = diff ** 2
        d = sum(diff)
        distances.append((i, d))

    # Sort distances
    distances = sorted(distances, key=lambda x: x[1])
    # Get the majority vote
    votes = defaultdict(int)
    for i in xrange(k):
        votes[classes[distances[k][0]]] += 1

    predicted_class = None
    for c, v in votes.iteritems():
        if predicted_class == None:
            predicted_class = (c, v)
            continue

        if predicted_class[1] < v:
            predicted_class = (c, v)

    return predicted_class[0]


if __name__ == '__main__':
    inputs, outputs = load_data([0, 1], 4, "../data/iris.csv")
    import sys
    c = classify([float(x) for x in sys.argv[1:]], inputs, outputs)
    print "Class: " + c
