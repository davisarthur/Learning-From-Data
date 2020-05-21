import numpy as np
import matplotlib.pyplot as plt
import random
import math

##
# Davis Arthur
# Recursive Legendre Polynomial Generator
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

# Generate and plot the first 6 Legendre polynomials
def partA(numPoints):
    # arrays to hold the x and y arrays of all 6 polynomials
    bigX = []
    bigY = []
    for i in range(6):
        # arrays to hold the x and y arrays of a Legendre polynomial
        X = []
        Y = []
        for j in range(numPoints + 1):
            deltaX = 2.0 / numPoints
            x = -1.0 + deltaX * j
            X.append(x)
            Y.append(legendrePolynomial(i, x))
        bigX.append(X)
        bigY.append(Y)

    # plot each polynomial
    colors = ["g", "r", "c", "m", "y", "k"]
    for i in range(6):
        plt.plot(bigX[i], bigY[i], colors[i], label = "$L_{" + str(i) + "}$")

    plt.title("Problem 4.3 A")
    plt.xlabel("$x$")
    plt.ylabel("$L_{q}$")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    partA(100)
