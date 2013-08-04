import sys
import numpy as np

def load_data(filename):
    inputs = []
    outputs = []
    with open(filename) as fd:
        for line in fd:
            tokens = line.strip().split(',')
            ins = [1.0] + [ float(x) for x in tokens[:-1] ]
            inputs.append(ins)
            outputs.append(float(tokens[-1]))
    return (inputs, outputs)


class LinearRegressionModel(object):
    def __init__(self, model=None):
        self.model = model

    def predict(self, *args):
        x = [1] + list(args)
        prediction = sum([x[i] * self.model.A[i][0] for i in xrange(len(x))])
        return prediction

    def fit(self, inputs, outputs):
        X = np.mat(inputs)
        Y = np.mat(outputs).T
        x_trans_x = X.T * X
        if np.linalg.det(x_trans_x) == 0:
            raise Exception('Cannot inverse singular matrix')
        self.model = x_trans_x.I * (X.T * Y)


if __name__ == '__main__':
    inputs, outputs = load_data("../data/house_prices.txt")
    lm = LinearRegressionModel()
    lm.fit(inputs, outputs)
    input_data = [float(x) for x in sys.argv[1:3]]
    print lm.predict(*input_data)
