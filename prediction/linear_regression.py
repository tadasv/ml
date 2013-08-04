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
    def __init__(self, inputs=None, outputs=None, model=None):
        self.model = model
        self.inputs = inputs
        self.outputs = outputs

    def sse(self):
        diff = np.mat(self.outputs).T - (self.inputs * self.model)
        return sum(diff.A.flatten()**2)

    def predict(self, *args):
        x = [1] + list(args)
        prediction = sum([x[i] * self.model.A[i][0] for i in xrange(len(x))])
        return prediction

    def coeffdet(self):
        """
        R squared
        """
        y_mean = np.mean(self.outputs)
        ss_total = sum([(self.outputs[i] - y_mean)**2 for i in xrange(len(self.outputs))])

        #predictions = self.inputs * self.model
        #ss_regression = sum((predictions - y_mean)**2)
        ss_residuals = self.sse()

        r_squared = 1 - (ss_residuals/ss_total)
        return r_squared

    def fit(self):
        X = np.mat(self.inputs)
        Y = np.mat(self.outputs).T
        x_trans_x = X.T * X
        if np.linalg.det(x_trans_x) == 0:
            raise Exception('Cannot inverse singular matrix')
        self.model = x_trans_x.I * (X.T * Y)


if __name__ == '__main__':
    inputs, outputs = load_data("../data/house_prices.txt")
    input_data = [float(x) for x in sys.argv[1:3]]
    lm = LinearRegressionModel(inputs=inputs, outputs=outputs)
    lm.fit()
    print 'Model: {0}'.format(lm.model)
    print 'Prediction: {0}'.format(lm.predict(*input_data))
    print 'R Squared: {0}'.format(lm.coeffdet())
