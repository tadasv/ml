"""
k nearest neighbours classifier
"""


import csv
import numpy as np
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


def classify(X, training_data, classes, k=3):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    distances = []
    for i, training_item in enumerate(training_data):
        diff = X - training_item
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


inputs, outputs = load_data([0, 1], 4, "../data/iris.csv")
import sys
c = classify([float(x) for x in sys.argv[1:]], inputs, outputs)
print "Class: " + c
