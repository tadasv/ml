"""
Locally weighted linear regression
"""


import csv
import numpy as np


def read_iris_data(filename):
    sepalWidths = []
    sepalLengths = []

    with open(filename) as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            # Widths will be our inputs, include intercept
            sepalWidths.append([1.0, float(row['sepalWidth'])])
            sepalLengths.append(float(row['sepalLength']))
    return sepalWidths, sepalLengths


def gaussian_kernel(x, x0, c, a=1.0):
    """
    Gaussian kernel.

    :Parameters:
      - `x`: nearby datapoint we are looking at.
      - `x0`: data point we are trying to estimate.
      - `c`, `a`: kernel parameters.

    :Returns:

    """
    diff = x - x0
    dot_product = diff * diff.T
    return a * np.exp(dot_product / (-2.0 * c**2))


def get_weights(training_inputs, datapoint, c=1.0):
    """
    :Parameters:
      - `training_inputs`: training data set the weights should be assigned to.
      - `datapoint`: data point we are trying to predict.
      - `c`: kernel function parameter

    :Returns:
      NxN weight matrix, there N is the size of the `training_inputs`.
    """
    x = np.mat(inputs)
    # Identity matrix to hold all of our input rows.
    n_rows = x.shape[0]
    weights = np.mat(np.eye(n_rows))
    for i in xrange(n_rows):
        weights[i, i] = gaussian_kernel(datapoint, x[i], c)

    return weights


def lwlr_predict(training_inputs, training_outputs, datapoint, c=1.0):
    """
    Predict a data point by fitting local regression.

    :Parameters:
      - `training_inputs`: training input data.
      - `training_outputs`: training outputs.
      - `datapoint`: data point we want to predict.
      - `c`: kernel parameter.

    :Returns:
      Estimated value at `datapoint`.
    """
    weights = get_weights(training_inputs, datapoint, c=c)

    x = np.mat(training_inputs)
    y = np.mat(training_outputs).T

    xt = x.T * (weights * x)
    if np.linalg.det(xt) == 0.0:
        raise Exception('Matrix is singular')

    betas = xt.I * (x.T * (weights * y))

    # prediction
    return datapoint * betas


inputs, outputs = read_iris_data("../data/setosa.csv")
X = [x[1] for x in inputs]

test_X = np.arange(min(X), max(X), 0.02)
predictions01 = []
predictions02 = []
predictions1 = []
# Create objects for javascript
js_predictions01 = []
js_predictions02 = []
js_predictions1 = []
for item in test_X:
    res01 = lwlr_predict(inputs, outputs, [1, item], c=0.1).A.flatten()[0]
    res02 = lwlr_predict(inputs, outputs, [1, item], c=0.2).A.flatten()[0]
    res1 = lwlr_predict(inputs, outputs, [1, item], c=1).A.flatten()[0]
    predictions01.append(res01)
    predictions02.append(res02)
    predictions1.append(res1)

    js_predictions01.append((item, res01))
    js_predictions02.append((item, res02))
    js_predictions1.append((item, res1))


doc = {
    'c_01': js_predictions01,
    'c_02': js_predictions02,
    'c_1': js_predictions1,
}

import json
print json.dumps(doc, indent=2)

from sys import exit
exit()

import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(test_X, predictions01, c='red')
ax.plot(test_X, predictions02, c='blue')
ax.plot(test_X, predictions1, c='green')
ax.scatter(X, outputs, s=2, c='black')
plt.show()
