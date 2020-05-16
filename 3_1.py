import numpy as np
import matplotlib.pyplot as plt
import random
import math

##
# Davis Arthur
# The perceptron algorithm
# Learning From Data — Problem 3.1
# 5-15-2020
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

# Perceptron algorithm from LFD
def perceptron(dimension, points):
    weights = np.zeros(dimension + 1)
    count = 0   # Number of iterations

    while True:
        updated = False
        count = count + 1

        for point in points:
            hValue = hypothesis(weights, point.genArray())
            if point.color == "B" and hValue == -1:
                weights = update(weights, point.genArray())
                updated = True
                break
            if point.color == "R" and hValue == 1:
                weights = update(weights, point.genArray())
                updated = True
                break

        if not updated:
            break

    return weights, count

def hypothesis(weightsIn, point):
    if np.dot(weightsIn, point) >= 0:
        return 1
    return -1

def update(prevW, pointArrIn):
    output = np.add(prevW, -hypothesis(prevW, pointArrIn) * pointArrIn)
    return output

# linear regression algorithm from LFD
def linreg(points):
    # create the input data matrix
    X = np.array()
    for point in points:
        X.append(point.genArray())

    print("X: " + X)

    # create the y-vector from sample points
    Y = np.array()
    for point in points:
        if point.color == "R":
            Y.append(1.0)
        else:
            Y.append(-1.0)

    # calculate the weights using regression formula
    psuedoInv = np.matmul(np.linalg.inv(np.matmul(np.transpose(X), X)), np.transpose(X))
    weights = np.matmul(psuedoInv, Y)
    return weights

def partAB(numPoints, radius, thk, sep):
    # Generate input data
    points = genPoints(numPoints, radius, thk, sep)

    # Plot input data
    for point in points:
        if point.color == "B":
            plt.scatter(point.x, point.y, c = "b")
        else:
            plt.scatter(point.x, point.y, c = "r")

    # Run PLA
    weights, count = perceptron(2, points)
    print("Number of iterations for PLA: " + str(count))

    # Plot PLA generated line
    plaLine = Line(weights)
    plaX, plaY = plaLine.producePoints(-radius - thk, radius * 2 \
        + thk / 2, 3)
    plt.plot(plaX, plaY, "g--", label = "Perceptron Learning Algorithm")

    # Run linear regression algorithm and plot generated line
    linregLine = Line(linreg(points))
    linRegX, linRegY = linRegLine.producePoints(-radius - thk, radius * 2 \
        + thk / 2, 3)
    plt.plot(linRegX, linRegY, "y--", label = "Linear Regression")

    plt.title("Problem 3.1")
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.legend()
    plt.show()

# Main method
if __name__ == "__main__":
    partAB(2000, 10.0, 5.0, 5.0)
