# Nick Smith
# CS 445 - Machine Learning
# HW #2
# Neural Networks
# Due Thursday January 31, 2019 - 5 pm


# ***************************** #
#                               #
#   PROGRAM TUNING VARIABLES    #
#                               #
# ***************************** #

# The Functionality of this program can be manipulated
# with the following variables:

# LEARNING RATE
lrn_rate = 0.1

# IMAGE SIZE
num_pixels = 28 * 28  # pixels

# Possible number outputs - 1 through 10
classes = 10

# Hidden units
hidden_nodes = 100

# Momentum Value
momentum_val = .9

# Training/Testing Values
# num_training = 60000
num_training = 15000
# num_training = 30000

# EPOCHS
num_epochs = 50

# initialize mini batch size
mini_batch_size = 16

# initialize each layer sizes as a list
# (input layer, hidden layer, output)
layers = [num_pixels, hidden_nodes, classes]

# ***************************** #
#                               #
#       PROGRAM FUNCTIONS       #
#                               #
# ***************************** #

# INCLUDED LIBRARIES
import gzip
import random
import numpy as np
import cPickle
import os
import matplotlib.pyplot as plt


# One hot number representation
def one_hot_result(y):
    e = np.zeros((10, 1))
    e[y] = 1.0
    return e


def mnist_loader():
    data_file = gzip.open(os.path.join(os.curdir, 'data', 'mnist.pkl.gz'), 'rb')
    training_data, validation_data, test_data = cPickle.load(data_file)
    data_file.close()

    #for weight training purposes
    training_inputs = [np.reshape(x, (784, 1)) for x in training_data[0]]
    training_results = [one_hot_result(y) for y in training_data[1]]
    training_data = zip(training_inputs, training_results)

    #to test the weights/graphical purposes
    validation_inputs = [np.reshape(x, (784, 1)) for x in validation_data[0]]
    validation_results = validation_data[1]
    validation_data = zip(validation_inputs, validation_results)

    #to test the algorithm
    test_inputs = [np.reshape(x, (784, 1)) for x in test_data[0]]
    test_data = zip(test_inputs, test_data[1])
    return training_data, test_data, validation_data


# Sigmoid and Sigmoid Prime functions
def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoidP(z):
    return sigmoid(z) * (1 - sigmoid(z))

# ***************************** #
#                               #
#        NEURAL NETWORK         #
#                               #
# ***************************** #

class NeuralNetwork(object):

    # ***** INITIALIZATIONS ***** #

    def __init__(self, learn_rate, momentum, mini_batch, epochs, sizes=list()):

        # sizes is a list that will decide how many hidden layers are present
        self.sizes = sizes
        self.num_layers = len(sizes)
        # assign random initial weights to input layer and hidden layer
        self.weights = [np.array([0])] + [np.random.randn(y, x)
                                          for y, x in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.randn(1, 1) for y in sizes]
        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]
        self.mini_batch_size = mini_batch
        self.epochs = epochs
        self.lrt = learn_rate
        self.momentum = momentum
        self.confusion_matrix = []

    # ***** STOCHASTIC GRADIENT DESCENT ***** #

    def SGD(self, training_data, validation_data, trn_data):
        trn_acc_array = []
        test_acc_array = []
        for epoch in range(self.epochs):

            #shuffle training examples
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.mini_batch_size] for k in
                range(0, len(training_data), self.mini_batch_size)]

            # batch updates increase the algorithm time quite a bit
            for mini_batch in mini_batches:
                dot_b = [np.zeros(bias.shape) for bias in self.biases]
                dot_w = [np.zeros(weight.shape) for weight in self.weights]
                for x, y in mini_batch:
                    self.forward_prop(x)
                    delta_dot_b, delta_dot_w = self.back_prop(x, y)
                    dot_b = [db + ddb*self.momentum for db, ddb in zip(dot_b, delta_dot_b)]
                    dot_w = [dw + ddw*self.momentum for dw, ddw in zip(dot_w, delta_dot_w)]

                self.weights = [
                    w - (self.lrt / self.mini_batch_size) * dw for w, dw in
                    zip(self.weights, dot_w)]
                self.biases = [
                    b - (self.lrt / self.mini_batch_size) * db for b, db in
                    zip(self.biases, dot_b)]

            trn_accuracy = self.validate_trn(trn_data)
            trn_accuracy /= 100.0
            print("Epoch {0}, Training Accuracy {1} %.".format(epoch + 1, trn_accuracy))
            trn_acc_array.append(trn_accuracy)

            tst_accuracy, self.confusion_matrix = self.validate_test(validation_data)
            tst_accuracy /= 100.0
            print("Epoch {0}, Test Accuracy {1} %.".format(epoch + 1, tst_accuracy))
            test_acc_array.append(tst_accuracy)


            np.set_printoptions(precision=2, threshold=100000,linewidth=2000, suppress=True)
            print(np.matrix(self.confusion_matrix))

        plt.plot(trn_acc_array, label="training", color='red')
        plt.plot(test_acc_array, label="testing", color='blue')
        plt.legend()
        plt.show()

    def validate_trn(self, validation_data):

        validation_results = [(self.predict(x) == y) for x, y in validation_data]
        return sum(result for result in validation_results)

    def validate_test(self, validation_data):

        self.confusion_matrix = np.zeros((10, 10))
        validation_results = [(self.predict(x) == y) for x, y in validation_data]
        for x, y in validation_data:
            if ((self.predict(x) == y).any()):
                self.confusion_matrix[self.predict(x), self.predict(x)] += 1
            else:
                self.confusion_matrix[self.predict(x), y] += 1

        return sum(result for result in validation_results), self.confusion_matrix



    def predict(self, x):

        self.forward_prop(x)
        return np.argmax(self._activations[-1])

    def forward_prop(self, x):
        self._activations[0] = x
        for i in range(1, self.num_layers):
            self._zs[i] = (
                    (self.weights[i].dot(self._activations[i - 1])) + self.biases[i]
            )
            self._activations[i] = sigmoid(self._zs[i])

    def back_prop(self, x, y):
        dot_b = [np.zeros(bias.shape) for bias in self.biases]
        dot_w = [np.zeros(weight.shape) for weight in self.weights]

        error = (self._activations[-1] - y) * sigmoidP(self._zs[-1])
        dot_b[-1] = error
        dot_w[-1] = error.dot(self._activations[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l + 1].transpose().dot(error),
                sigmoidP(self._zs[l])
            )
            dot_b[l] = error
            dot_w[l] = error.dot(self._activations[l - 1].transpose())

        return dot_b, dot_w


# Run the program stuff

# initialize training, validation and testing data
training_data, test_data, validation_data = mnist_loader()

# initialize neuralnet
nn = NeuralNetwork(lrn_rate, momentum_val, mini_batch_size, num_epochs, layers)

# training neural network
nn.SGD(training_data, test_data, validation_data)
