"""
k nearest neighbours classifier
"""


import sys
import csv
import numpy as np
from random import choice
from collections import defaultdict
from operator import itemgetter

from sklearn.neighbors import KNeighborsClassifier


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

    return np.array(inputs), np.array(outputs)


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
    # generated using sample_indices
    sample = [116, 139, 51, 43, 56, 98, 5, 23, 16, 11, 135, 122, 70, 82, 80]
    rest = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20, 21,
            22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41,
            42, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 78, 79, 81, 83, 84, 85,
            86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103,
            104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 118, 119,
            120, 121, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 136,
            137, 138, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149]

    inputs, outputs = load_data([0, 1], 4, "../data/iris.csv")

    # Slice data set into test and training data
    sample_idx = np.array(sample)
    rest_idx = np.array(rest)
    test_in = inputs[sample_idx]
    test_out = outputs[sample_idx]
    rest_in = inputs[rest_idx]
    rest_out = outputs[rest_idx]

    #test_data = [float(x) for x in sys.argv[1:]]
    #c = classify(test_data, inputs, outputs)
    #print "Class: " + c

    #print "Scikit"
    total_items = len(sample)

    json_doc = {
        'test_data': {
            'in': test_in.tolist(),
            'out': test_out.tolist(),
        },
        'predictions': {
        }
    }

    neighbours = 1
    while neighbours <= int(sys.argv[1]):
        knn = KNeighborsClassifier(n_neighbors=neighbours)
        knn.fit(rest_in, rest_out)
        predictions = knn.predict(test_in)
        json_doc['predictions'][neighbours] = predictions.tolist()
        invalid_results = 0
        for i, prediction in enumerate(predictions):
            if test_out[i] != prediction:
                invalid_results += 1

        print 'neighbours {0}, error rate {1}'.format(neighbours, float(invalid_results)/total_items)
        neighbours += 1

    import json
    print json.dumps(json_doc, indent=2)
