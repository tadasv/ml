from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from linear_regression import cost_derivative
from linear_regression import hypothesis


class GradientDescent(object):
    def __init__(self, rate=0.1, error_threshold=0.0001, function=None):
        # Number of iterations it took to fit the data
        self.num_iter = 0
        self.error_threshold = error_threshold
        # Learning rate (alpha)
        self.rate = rate
        self.coef = None
        # Function to run gradient descent on
        self.function = function

    def _done(self, deltas):
        n = len(deltas)
        delta = [ abs(deltas[i]) <= self.error_threshold for i in xrange(n) ]
        return all(delta)

    def predict(self, input_values):
        return self.function(self.coef, input_values)

    def fit(self, inputs, outputs):
        num_variables = len(inputs)
        self.num_iter = 0
        self.coef = [0] * num_variables

        while True:
            self.num_iter += 1
            thetas = cost_derivative(self.coef, inputs, outputs)
            # Steps
            deltas = [t * self.rate for t in thetas]
            for i in xrange(len(deltas)):
                self.coef[i] = self.coef[i] - deltas[i]

            if self._done(deltas):
                return


def test(inputs, outputs):
    gd = GradientDescent(function=hypothesis)
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
