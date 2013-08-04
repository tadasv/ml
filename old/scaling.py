from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def scale(input_values):
    num_inputs = len(input_values)
    num_variables = len(input_values[0])

    scaled = [ [0] * num_variables for _ in xrange(num_inputs) ]

    for j in xrange(num_variables):
        min_ = None
        max_ = None
        s = 0

        # Get avg and range
        for i in xrange(num_inputs):
            value = input_values[i][j]
            min_ = min(value, min_ or value)
            max_ = max(value, max_ or value)
            s += value

        avg = float(s) / num_inputs
        range_ = max_ - min_

        # Scale
        for i in xrange(num_inputs):
            value = input_values[i][j]
            if range_ != 0:
                ss = (value - avg) / range_
            else:
                # NOTE don't scale the bias
                ss = value

            scaled[i][j] = ss

    return scaled


def main():
    inputs = [ [1, 4],
               [2, -4],
               [1, 555],
             ]

    print("Original:")
    print(inputs)
    print("Scaled:")
    print(scale(inputs))

if __name__ == '__main__':
    main()
