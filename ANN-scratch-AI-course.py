# AI course
# ANN from scratch
# dummy data
# William Henriksson

# these are the librarys needed to run this script
import sklearn.datasets
import numpy as np
import matplotlib.pyplot as plt

# builds a network with specified number of units
# for this assignment use build_network(2,4,1)
def build_network(num_inputs, num_hidden, num_outputs):
    np.random.seed(42)
    W1 = np.random.rand(num_inputs, num_hidden)*0.01
    B1 = np.zeros((1,num_hidden))
    W2 = np.random.rand(num_hidden, num_outputs)*0.01
    B2 = np.zeros((1,num_outputs))
    network = { 'W1': W1, 'B1': B1, 'W2': W2, 'B2': B2}
    return network


# The logistic function.
# Should return a vector/matrix with the same dimensions as x.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# the derivative of the activation function
# implementing an using this might lead to a more readable code
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# calculate networks output given input X
# returns for each sample in X the corresponding network's output y_hat
def forward_propagate(network, X):
    z1 = X.dot(network["W1"]) + network["B1"]
    a1 = sigmoid(z1)

    z2 = a1.dot(network["W2"]) + network["B2"]
    y_hat = sigmoid(z2) # sigmoid(z2) is the same as "a2"

    return (z1, a1, z2, y_hat)


# evaluate cost over a whole data set
# this will be computed during training to track progress
def cost(network, dataset):
    X = dataset['X']
    Y = dataset['Y']

    f_result = forward_propagate(network, X)
    y_hat = f_result[3]

    mse = np.square(y_hat - Y).mean()
    return mse


def train(network, dataset, max_epochs):
    # the dataset is a list of inputs and targets
    # and is accessed like this
    X = dataset['X']
    Y = dataset['Y']

    for epoch in np.arange(0,max_epochs):

        # ###
        # Forward propagation
        # ###
        
        # returns multiple values, input and output of the hidden layer and output unit
        f_result = forward_propagate(network, X)

        z1 = f_result[0]
        a1 = f_result[1]
        z2 = f_result[2]
        y_hat = f_result[3]

        # ###
        # Backpropagation
        # ###

        # implement the code to compute the parameter updates here
        # includeing the bias update

        mse = cost(network, dataset)

        # changing the format of the Y
        newY = []
        for i in Y:
            newY.append([i])
        # derivative of cost
        dc = (y_hat - newY)

        # # derivatives
        dz1 = d_sigmoid(z1)
        da1 = d_sigmoid(a1)
        dz2 = d_sigmoid(z2)        

        # # calculation of delta Bias 2
        db2 = dz2 * dc
        print(db2.shape, 'db2')


        # # calculation of delta B1
        db1 = dz1 * dc
        print(db1.shape, 'db1')
        #New stuff
        #output error = cost diff and derv z2
        outputError = dc * dz2
        print(outputError.shape, 'output error')

        #cost weight deriv - most likely not da1
        dW2 = outputError * da1/a1
        print(dW2.shape, 'dw2')
        print(network['W2'].shape, 'N W2')
        #hiddenlayer error
        HiddenError = outputError * np.transpose(network["W2"]) * dz1

        #cost weight deriv
        dW1 = np.transpose(HiddenError) * X
        print(network['W1'].shape, 'N W1')
        print(dW1.shape, 'dW1')
        # ###
        # Update the networks parameters here
        # ###

        alpha = 0.03 #learning rate
        network['W1'] = network['W1'] - (dW1 * alpha)
        network['B1'] = network['B1'] - (db1 * alpha)
        network['W2'] = network['W2'] - (np.transpose(dW2) * alpha)
        network['B2'] = network['B2'] - (db2 * alpha)

        # some printouts
        # so we have some idea what the network is doing
        if (epoch%100 == 0):
            cost_temp = cost(network, dataset)
            print('>epoch=%d, cost=%.6f' % (epoch, cost_temp))

    # returning the trained network to plot results
    return network

# returns the from the network assigned classes to the data given in X
# returned classes are either zero or one
def classify(network, X):
    f_result = forward_propagate(network, X)
    return np.round(f_result[3])

##### ##### #### ##### ##### #####
##### ##### DATA ##### ##### #####
##### ##### #### ##### ##### #####

# normalizing each column of x
def normalize(x):
    return (x - np.mean(x,0)) / np.std(x,0)

# BLOBS data
X, Y = sklearn.datasets.make_blobs(n_samples=400, centers=2, n_features=2,random_state=0)
X = normalize(X)
dataset_blobs = {'X': X, 'Y': Y}
# print(dataset_blobs)


##### ##### #### ##### ##### #####
##### ##### PLOT ##### ##### #####
##### ##### #### ##### ##### #####
def plot_summary(net, dataset):
    # area to be evaluated
    maxs = dataset['X'].max(axis=0)*1.1
    mins = dataset['X'].min(axis=0)*1.1
    x1 = np.linspace(mins[0],maxs[0], 400)
    x2 = np.linspace(mins[1], maxs[1], 400)
    x1v, x2v = np.meshgrid(x1,x2)
    # predict classes
    Z = classify(net,np.dstack(np.meshgrid(x1,x2)))
    Z = Z.reshape(len(x1),len(x2))
    # Plot the contour and training examples
    plt.contourf(x1v, x2v, Z, cmap='Paired')
    plt.scatter(dataset['X'][:, 0], dataset['X'][:, 1], c=dataset['Y'], cmap='Paired')
    plt.title("Network's Decision Landscape")
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

##### ##### ### ##### ##### #####
##### ##### RUN ##### ##### #####
##### ##### ### ##### ##### #####

# this will be executed upon calling the script

network = build_network(2, 4, 1)

trainset = dataset_blobs 
# train(network, trainset, 10)

network = train(network, trainset, 10)
plot_summary(network, trainset)

