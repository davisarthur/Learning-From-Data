import numpy as np
import matplotlib.pyplot as plt
import random
import math

##
# Davis Arthur
# Pocket Algorithm (adaptation of Perceptron Learning Algortithm)
# Learning From Data — Problem 3.3
# 5-16-2020
##

# Class used to represent a 2D point
class Point:

    def __init__(self, xIn, yIn, colorIn = None):
        self.x = xIn
        self.y = yIn
        self.color = colorIn

    def genArray(self):
        return np.array([1.0, self.x, self.y])

# Class used to represent a 2D line
class Line:

    def __init__(self, weightsIn):
        self.weights = weightsIn

    def above(self, pointIn):
        return np.dot(self.weights, pointIn) >= 0

    def producePoints(self, minX, maxX, numPoints):
        x = []
        y = []
        interval = (maxX - minX) / (numPoints - 1)
        for i in range(numPoints):
            x.append(minX + i * interval)
            y.append(float(-self.weights[1] * x[-1] - self.weights[0]) \
                / float(self.weights[2]))
        return x, y

def genPoints(numPoints, radius, thickness, sep):
    points = []
    for i in range(numPoints):
        r = random.uniform(radius, radius + thickness)
        angle = random.random() * np.pi
        topRing = random.random() > 0.5
        if topRing:
            points.append(Point(r * math.cos(angle), r * math.sin(angle), "R"))
        else:
            points.append(Point(r * math.cos(-angle) + radius + thickness / 2.0, \
                r * math.sin(-angle) - sep, "B"))
    return points

# Pocket algorithm based on PLA
def pocket(dimension, points, numItr):
    weights = np.zeros(dimension + 1)

    for i in range(numItr):
        updated = False

        for point in points:
            hValue = hypothesis(weights, point.genArray())
            if point.color == "B" and hValue == -1:
                newWeights = update(weights, point.genArray())
                prevErr, newErr = calcErrors(weights, newWeights, points)
                if newErr < prevErr:
                    weights = newWeights
                updated = True
                break
            if point.color == "R" and hValue == 1:
                newWeights = update(weights, point.genArray())
                prevErr, newErr = calcErrors(weights, newWeights, points)
                if newErr < prevErr:
                    weights = newWeights
                updated = True
                break

        if not updated:
            break

    return weights

# linear regression algorithm from LFD
def linreg(points):
    # create the input data matrix
    X = np.zeros((len(points), 3))
    i = 0
    for point in points:
        X[i] = point.genArray()
        i = i + 1

    # create the y-vector from sample points
    Y = np.zeros(len(points))
    i = 0
    for point in points:
        if point.color == "R":
            Y[i] = 1.0
        else:
            Y[i] = -1.0
        i = i + 1

    # calculate the weights using regression formula
    psuedoInv = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X))
    weights = np.matmul(psuedoInv, Y)
    return weights

def calcErrors(prevWeights, newWeights, points):
    prevErr = 0
    newErr = 0
    for point in points:
        prevH = hypothesis(prevWeights, point.genArray())
        if point.color == "B" and prevH == -1:
            prevErr = prevErr + 1
        if point.color == "R" and prevH == 1:
            prevErr = prevErr + 1
        newH = hypothesis(newWeights, point.genArray())
        if point.color == "B" and newH == -1:
            newErr = newErr + 1
        if point.color == "R" and newH == 1:
            newErr = newErr + 1
    return prevErr, newErr

def hypothesis(weightsIn, point):
    if np.dot(weightsIn, point) >= 0:
        return 1
    return -1

def update(prevW, pointArrIn):
    output = np.add(prevW, -hypothesis(prevW, pointArrIn) * pointArrIn)
    return output

def partAB(numPoints, radius, thk, sep, numItr):
    # Generate input data
    points = genPoints(numPoints, radius, thk, sep)

    # Plot input data
    for point in points:
        if point.color == "B":
            plt.scatter(point.x, point.y, c = "b")
        else:
            plt.scatter(point.x, point.y, c = "r")

    # Run pocket algorithm
    weights = pocket(2, points, numItr)

    # Plot pocket algorithm generated line
    pocketLine = Line(weights)
    pocketX, pocketY = pocketLine.producePoints(-radius - thk, radius * 2 \
        + thk * 2, 2)
    plt.plot(pocketX, pocketY, "g--", label = "Pocket Algorithm")

    # Run linear regression algorithm and plot generated line
    linRegLine = Line(linreg(points))
    linRegX, linRegY = linRegLine.producePoints(-radius - thk, radius * 2 \
        + thk * 2, 2)
    plt.plot(linRegX, linRegY, "y--", label = "Linear Regression")

    plt.title("Problem 3.3")
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.legend(loc = "lower left")
    plt.show()

# Main method
if __name__ == "__main__":
    partAB(1000, 10.0, 5.0, -1.0, 10000)
