import numpy as np
import matplotlib.pyplot as plt
import random
import math

##
# Davis Arthur
# Linear Regression and Overfitting
# Learning From Data — Problem 4.4
# 5-20-2020
##

# k — order
# x - variable
def legendrePolynomial(k, x):
    if k == 0:
        return 1
    if k == 1:
        return x
    return (2.0 * k - 1.0) / k * x * legendrePolynomial(k - 1, x) \
        - (k - 1.0) / k * legendrePolynomial(k - 2, x)

# Generates a random target function normalized so pow(f, 2) = 1
# qf - degree
# numNom - number of points used in normalization
def randTarget(qf, numNorm):
    coefficients = []
    for i in range(qf + 1):
        coefficients.append(np.random.normal())

    # Approximate the integral of pow(f, 2) along interval [-1, 1]
    deltaX = 2.0 / numNorm
    integralf2 = 0.0
    for i in range(numNorm):
        xi = -1.0 + i * deltaX
        fi = 0.0
        for j in range(qf + 1):
            fi = fi + coefficients[j] * legendrePolynomial(j, xi)
        fi2 = fi ** 2.0
        integralf2 = integralf2 + fi2 * deltaX

    # Normalize the coefficients
    expectedf2 = integralf2 / 2.0
    normalizationFactor = 1.0 / expectedf2
    for co in coefficients:
        co = co * normalizationFactor

    return coefficients

# Generate points
# qf — degree of target function
# N — number of points
# sigma - std deviation
def genPoints(qf, N, sigma, numNorm):
    # generate target function coefficients
    coefficients = randTarget(qf, numNorm)
    pointsX = []
    pointsY = []

    for i in range(N):
        x = random.uniform(-1.0, 1.0)
        pointsX.append(x)
        y = 0
        for j in range(qf + 1):
            y = y + coefficients[j] * legendrePolynomial(j, x)
        epsilon = np.random.normal()
        y = y + sigma * epsilon
        pointsY.append(y)

    return coefficients, pointsX, pointsY

# linear regression algorithm (non-linear transformation)
# pointsX - array containing the x coordinate of each input point
# pointsY - array containing the y coordinate of each input point
# degree — degree of Legendre polynomial used to generate hypothesis
def linreg(pointsX, pointsY, degree):
    # create the input data matrix
    Z = np.zeros((len(pointsX), degree + 1))
    for i in range(len(pointsX)):
        z = []
        for j in range(degree + 1):
            z.append(legendrePolynomial(j, pointsX[i]))
        Z[i] = np.array(z)

    # create the y-vector from sample points
    y = np.zeros(len(pointsY))
    for i in range(len(pointsY)):
        y[i] = pointsY[i]

    # calculate the weights using regression formula
    psuedoInv = np.matmul(np.linalg.inv(np.matmul(np.transpose(Z), Z)), np.transpose(Z))
    weights = np.matmul(psuedoInv, y)
    return weights

# calculate the error between hypothesis function and target function
# h - array of hypothesis values
# f - array of corresponding target function values
def error(h, f):
    error = 0.0
    for i in range(len(h)):
        error = error + (h[i] - f[i]) ** 2.0
    return error / len(h)

def part(qf, N, sigma, numTargetPoints, titleAdd):
    coefficients, pointsX, pointsY = genPoints(qf, N, sigma, numTargetPoints)

    # plot the target function
    deltaX = 2.0 / numTargetPoints
    targetX = []
    targetY = []
    for i in range(numTargetPoints + 1):
        x = -1.0 + i * deltaX
        y = 0.0
        for j in range(qf + 1):
            y = y + coefficients[j] * legendrePolynomial(j, x)
        targetX.append(x)
        targetY.append(y)
    plt.plot(targetX, targetY, "y", label = "$f_{" + str(qf) + "}$")

    # plot input data
    for i in range(len(pointsX)):
        plt.scatter(pointsX[i], pointsY[i], c = "b")

    # calculate 1st hypothesis function (Legendre polynomial of order 2)
    h2weights = linreg(pointsX, pointsY, 2)
    h2X = []
    h2Y = []
    for i in range(numTargetPoints + 1):
        x = -1.0 + i * deltaX
        y = 0.0
        for j in range(2 + 1):
            y = y + h2weights[j] * legendrePolynomial(j, x)
        h2X.append(x)
        h2Y.append(y)
    plt.plot(h2X, h2Y, "g", label = "$g_2$")

    # calculate 1st hypothesis function (Legendre polynomial of order 10)
    h10weights = linreg(pointsX, pointsY, 10)
    h10X = []
    h10Y = []
    for i in range(numTargetPoints + 1):
        x = -1.0 + i * deltaX
        y = 0.0
        for j in range(10 + 1):
            y = y + h10weights[j] * legendrePolynomial(j, x)
        h10X.append(x)
        h10Y.append(y)
    plt.plot(h10X, h10Y, "m", label = "$g_{10}$")

    # calculate error of h2 and h10
    h2error = error(h2Y, targetY)
    h10error = error(h10Y, targetY)
    print("g2 error: " + str(h2error))
    print("g10 error: " + str(h10error))
    print()

    # plot settings
    plt.title("Problem 4.4: " + titleAdd)
    plt.xlabel("$x$")
    plt.ylabel("$f(x)$")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Few points, 10th degree target function (g2 usually has lower error)
    part(10, 20, 0.1, 200, "Few Points, Low Noise")
    # Many points, 10th degree target function (g10 has much lower error)
    part(10, 120, 0.1, 200, "Many Points, Low Noise")
    # Many points, 10th degree target function, high noise (comparable error)
    part(10, 120, 1.0, 200, "Many Points, High Noise")
    # 25th degree target function, low noise (comparable error)
    part(25, 50, 0.1, 200, "Many Points, Low Noise")
    # 25th degree target function, high noise (g2 has lower error)
    part(25, 50, 0.5, 200, "Many Points, High Noise")
