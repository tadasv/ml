"""
Linear regression
"""


def hypothesis(thetas, input_values):
    """
    Multivariate linear regression hypothesis function

    h(X) = theta0*x0 + theta1*x1 + ... + thetan*xn

    @param coefficients: list, thetas
    @param input: list, x's
    """
    n = len(thetas)
    return sum([thetas[i] * input_values[i] for i in xrange(n)])


def cost_derivative(thetas, inputs, outputs):
    """
    Derivative of a linear regression cost function to be used in gradient
    descent.
    """
    num_variables = len(inputs[0])
    num_inputs = len(inputs)
    new_thetas = [0] * num_variables

    for j in xrange(num_variables):
        s = 0
        for i in xrange(num_inputs):
            s += 0.5 * (hypothesis(thetas, inputs[i]) - outputs[i]) * inputs[i][j]

        s = s * 1.0/num_inputs
        new_thetas[j] = s

    return new_thetas
