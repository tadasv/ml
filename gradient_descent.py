from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def linear_regression(coefficients, input_values):
    """
    Multivariate linear regression hypothesis function

    h(X) = t0x0 + t1x1 + ... + tnxn

    @param coefficients: list, T
    @param input_values: list, X
    """
    n = len(coefficients)
    return sum([coefficients[i] * input_values[i] for i in xrange(n)])


class GradientDescent(object):
    def __init__(self, rate=0.1, error_threshold=0.0001, function=None):
        # Number of iterations it took to fit the data
        self.num_iter = 0
        self.error_threshold = error_threshold
        # Learning rate (alpha)
        self.rate = rate
        self.coef = None
        # Function to run gradient descent on
        self.function = function or linear_regression

    def _done(self, deltas):
        n = len(deltas)
        delta = [ abs(deltas[i]) <= self.error_threshold for i in xrange(n) ]
        return all(delta)

    def predict(self, input_values):
        return self.function(self.coef, input_values)

    def fit(self, inputs, outputs):
        num_inputs = len(inputs)
        num_variables = len(inputs[0])
        # Model parameters
        self.coef = [0] * num_variables
        # Step sizes
        deltas = [0] * num_variables

        self.num_iter = 0
        while True:
            self.num_iter += 1
            for j in xrange(num_variables):
                # Cost function
                s = 0
                for i in xrange(num_inputs):
                    s += (self.predict(inputs[i]) - outputs[i]) * inputs[i][j]
                s = s * 1.0/(2*num_inputs)

                # Steps
                deltas[j] = self.rate * s
                self.coef[j] = self.coef[j] - deltas[j]

            if self._done(deltas):
                return


def test(inputs, outputs):
    gd = GradientDescent()
    errors = [0.00000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]
    for error in errors:
        gd.error_threshold = error
        print("Fitting data. rate:{0} error_threshold:{1}".format(gd.rate, gd.error_threshold))
        gd.fit(inputs, outputs)
        print("Took {0} iterations".format(gd.num_iter))
        print("Back testing:")
        for i in inputs:
            print(i, ":", gd.predict(i))


def main():
    inputs = [ [1, 1, 4],
               [1, 2, 5],
               [1, 3, 6],
             ]
    output = [ 3,
               2,
               1,
             ]

    print("## NON SCALED ##")
    test(inputs, output)
    print("## SCALED ##")
    from scaling import scale
    scaled = scale(inputs)
    test(scaled, output)

if __name__ == '__main__':
    main()
