import numpy as np

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

# learn pattern of data set and build a synapse
for iter in xrange(10000):
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

print "weights on inputs: %s \n" % nonlinear(syn0)

while(True):
    user_input = raw_input('Enter 3 numbers seperated by spaces: ').split(' ')
    input_arr = [int(num) for num in user_input]

    guess = round(nonlinear(np.dot(input_arr, syn0)))

    print ("guess: %s" % guess)
