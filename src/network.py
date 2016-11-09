#!/usr/bin/env python

import numpy as np
from . import args


def startNetwork(self):
    verbose = self.args["verbose"]
    iterations = self.args["iterations"]
    # defalt value of iterations
    if (not iterations):
        iterations = 10000

    if (verbose):
        print ("Iterations : %d" % iterations)

    # sigmoid function
    def nonlinear(x, derive=False):
        if (derive):
            return (x * (1 - x))
        return 1 / (1 + np.exp(-x))

    # input dataset
    X = np.array([[1, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [0, 0, 1],
                  [1, 1, 1]])

    # output dataset
    y = np.array([[1, 0, 1, 0, 1]]).T

    np.random.seed(1)

    syn0 = 2 * np.random.random((3, 1)) - 1

    if (verbose):
        print ("Random weight start values: %s \n" % syn0)

    # learn pattern of data set and build a synapse
    for iter in xrange(iterations):
        l0 = X
        l1 = nonlinear(np.dot(l0, syn0))

        l1_error = y - l1

        l1_delta = l1_error * nonlinear(l1, True)

        syn0 += np.dot(l0.T, l1_delta)

    i = 0
    while (i < len(l1)):
        l1[i] = round(l1[i])
        print ("input: %s, output: %d, guess: %d" % (X[i], y[i], l1[i]))
        i += 1

    if (verbose):
        print ("Weights after training: %s \n" % nonlinear(syn0))

    while (True):
        user_input = raw_input('Enter 3 numbers seperated by spaces to test training: ').split(' ')
        input_arr = [int(num) for num in user_input]

        guess = round(nonlinear(np.dot(input_arr, syn0)))

        print ("guess: %s" % guess)
