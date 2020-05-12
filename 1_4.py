import numpy as np
import matplotlib.pyplot as plt
import random

##
# Davis Arthur
# The perceptron algorithm
# Learning From Data — Problem 1.4
# 5-12-2020
##

# Class used to represent a 2D point
class Point:

    def __init__(self, xIn, yIn):
        self.x = xIn
        self.y = yIn
        self.color = None

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

def genRandomPoints(numPoints, max):
    output = []
    for i in range(numPoints):
        x = (random.random() - 0.5) * 2 * max
        y = (random.random() - 0.5) * 2 * max
        output.append(Point(x, y))
    return output

def genRandomLine(max):
    w0 = (random.random() - 0.5) * 2 * max
    w1 = (random.random() - 0.5) * 2 * max
    w2 = (random.random() - 0.5) * 2 * max
    return Line(np.array([w0, w1, w2]))

# Perceptron algorithm from LFD
def perceptron(dimension, points):
    weights = np.zeros(dimension + 1)
    count = 0

    while True:
        updated = False
        count = count + 1

        for point in points:
            hValue = hypothesis(weights, point.genArray())
            if point.color == "G" and hValue == -1:
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

# Update the weights used in perceptron algorithm
def update(prevW, pointArrIn):
    output = np.add(prevW, -hypothesis(prevW, pointArrIn) * pointArrIn)
    return output

def hypothesis(weightsIn, point):
    if np.dot(weightsIn, point) >= 0:
        return 1
    return -1

def partABCD(part, numPoints = 20, max = 100):
    line = genRandomLine(max)
    points = genRandomPoints(numPoints, max)

    # Distinguish between points using target function
    greenPoints = []
    redPoints = []
    for point in points:
        if line.above(point.genArray()):
            point.color = "G"
            greenPoints.append(point)
        else:
            redPoints.append(point)
            point.color = "R"

    # Plot target function
    lineX, lineY = line.producePoints(-max, max, 10)
    plt.plot(lineX, lineY, "b", label = "Target Function")

    # Plot green points
    for point in greenPoints:
        plt.scatter(point.x, point.y, c = "g")

    # Plot red points
    for point in redPoints:
        plt.scatter(point.x, point.y, c = "r")

    # Plot perceptron generated line
    learnedWeights, count = perceptron(2, points)
    print("Number of iterations of Perceptron: " + str(count))
    learnedLine = Line(learnedWeights)
    learnedX, learnedY = learnedLine.producePoints(-max, max, 10)
    plt.plot(learnedX, learnedY, "y--", label = "Hypothesis Function")

    plt.title("Problem 1.4 Part " + str(part))
    plt.xlabel("x-position")
    plt.ylabel("y-position")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    partABCD("B", 20)
    partABCD("C", 20)
    partABCD("D", 100)
