import numpy as np
import matplotlib.pyplot as plt
import random
import math

##
# Davis Arthur
# Regularization and Cross Validation
# Learning From Data — Problem 4.24
# 6-1-2020
##

# Note each point begins with a 1 to account for bias weight (weights[0])
# n - number of points
# d - dimension
def genPoints(n, d):
    points = np.zeros((n, d + 1))
    for i in range(n):
        newPoint = np.zeros(d + 1)
        newPoint[0] = 1
        for j in range(d):
            newPoint[j + 1] = np.random.normal()
        points[i] = newPoint
    return points

# d - dimension of input data
def genWeights(d):
    weights = np.zeros(d + 1)
    for i in range(d + 1):
        weights[i] = np.random.normal()
    return weights

# Generates a y array based on input points and weights
# var - noise variance
def genY(points, weights, var):
    Y = np.zeros(np.shape(points)[0])
    for i in range(np.shape(points)[0]):
        Y[i] = np.dot(weights, points[i]) + var * np.random.normal()
    return Y

# n - number of points
def regParam(n):
    return 0.05 / n

# linear reggresion with weight decay regularization
def linReg2(points, Y, regParam):
    invTerm = np.linalg.inv(np.matmul(np.transpose(points), points) \
        + regParam * np.identity(np.shape(points)[1]))
    weightsReg = np.matmul(np.matmul(invTerm, np.transpose(points)), Y)
    return weightsReg

# calculate cross validation error for a given index
# index - index being ignored
# inputX - original input data X
# inputY - original input data Y
def cvErrIndex(index, inputX, inputY):
    otherX = np.zeros((inputX.shape[0] - 1, inputX.shape[1]))
    otherY = np.zeros(inputY.shape[0] - 1)
    for i in range(inputX.shape[0] - 1):
        count = 0
        if i != index:
            otherX[count] = inputX[i]
            otherY[count] = inputY[i]
            count = count + 1
    hWeights = linReg2(otherX, otherY, regParam(inputX.shape[0] - 1))
    return (np.dot(hWeights, inputX[index]) - inputY[index]) ** 2.0

# calculate total cross validation estimate
# inputX - original input data X
# inputY - original input data Y
# hweights - hypothesis weights
def cvErr(inputX, inputY, hweights, regParam):
    cvErr = 0.0
    H = np.matmul(np.matmul(inputX, np.linalg.inv(np.matmul(np.transpose(inputX), inputX) \
        + regParam * np.identity(np.shape(inputX)[1]))), np.transpose(inputX))
    for i in range(np.shape(inputX)[0]):
        cvErr = cvErr + ((np.dot(hweights, inputX[i]) - inputY[i]) \
            / (1 - H[i][i])) ** 2.0
    return cvErr / np.shape(inputX)[0]

# n - number of points
# d - dimension of input data
# var - noise variance
def test(n, d, var):
    # Generate data
    inputX = genPoints(n, d)
    print("Input X: " + str(inputX))
    targetWeights = genWeights(d)
    inputY = genY(inputX, targetWeights, var)
    print("Input Y: " + str(inputY) + "\n\n")

    # Use linear regression with weight decay to estimate target weights
    weightsReg = linReg2(inputX, inputY, regParam(n))
    print("Target Weights: " + str(targetWeights))
    print("Hypothesis Weights: " + str(weightsReg))

    # Calculate e1
    e1 = cvErrIndex(0, inputX, inputY)
    print("e1: " + str(e1))

    # Calculate e2
    e2 = cvErrIndex(1, inputX, inputY)
    print("e2: " + str(e2))

    # Calculate Ecv
    ecv = cvErr(inputX, inputY, weightsReg, regParam(n))
    print("Ecv: " + str(ecv))

# numExperiments - number of experiments
# n - number of points
# d - dimension of input data
# var - noise variance
def partE(numExperiments, n, d, var):
    E1 = np.zeros(numExperiments)
    Ecv = np.zeros(numExperiments)

    # Run experiments
    for i in range(numExperiments):
        # Generate data
        inputX = genPoints(n, d)
        targetWeights = genWeights(d)
        inputY = genY(inputX, targetWeights, var)

        # Use linear regression with weight decay to estimate target weights
        weightsReg = linReg2(inputX, inputY, regParam(n))

        # Calculate e1
        E1[i] = cvErrIndex(0, inputX, inputY)

        # Calculate Ecv
        Ecv[i] = cvErr(inputX, inputY, weightsReg, regParam(n))

    print("Neff / N: " + str(np.std(E1) / np.std(Ecv) / n))

if __name__ == "__main__":
    # test(3 + 115, 3, 0.5)
    partE(50, 3 + 115, 3, 0.5)
