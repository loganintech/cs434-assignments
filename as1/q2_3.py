import sys
import numpy as np
import math
from mpmath import mp
import csv
import matplotlib.pyplot as plt


def loadCSV(fileName, nFeat):
    xload = []
    yload = []
    temp = []

    with open(fileName) as f:
        string = f.read().replace('\n', ',')
        lines = string.split(',')  # lines = list of every data point

    lines = lines[:-1]

    for count in range(0, len(lines)):
        if (count+1) % (nFeat+1) == 0:  # if position is 14(y value) append to y
            yload.append([float(lines[count])])
        else:  # else (if first position append a dummy '1') append to temp
            temp.append(float(lines[count]))

    # splitting x into a list of lists separated by each instance
    xload = [temp[i:i+nFeat] for i in range(0, len(temp), nFeat)]
    x1 = np.matrix(xload)  # converting x from list of lists to matrices
    y1 = np.matrix(yload)  # converting y from list of lists to matrices
    x1 = x1 * (1.0/255.0)  # converts x values into percentages

    return x1, y1


def sigmoid(w, x):
    # exponent = (-1 * np.dot(w, x.T))
    exponent = (-w.T) * x
    new_exponent = float(exponent.item(0))
    try:
        y_hat = 1./(1. + math.exp(new_exponent))
    except Exception as e:
        print(exponent)
        print(new_exponent)
        raise e
    return y_hat


def check_accuracy(w, x, y):
    correct = 0

    for i in range(0, x.shape[0]):
        if y[i] == 1:
            sig = sigmoid(w, x[i])
            if sig >= 0.5:
                correct += 1
        elif y[i] == 0:
            sigmoid_complement = 1 - sigmoid(w, x[i])
            if sigmoid_complement >= 0.5:
                correct += 1

    percentage_correct = correct / x.shape[0]
    return percentage_correct


def gradient(x, y, x1, y1, reg_strength=None, iterations=100):
    # **************************************************************************
    # LOGISTIC REGRESSION
    # **************************************************************************
    training_accuracies = []
    testing_accuracies = []

    if reg_strength is not None:
        try:
            reg_strength = float(reg_strength)
        except:
            reg_strength = None

    w = np.matrix([0] * 256)

    learning_rate = 0.001  # let the learning rate be 0.001

    for _ in range(iterations):
        gradient = np.matrix([0] * 256)
        for i in range(0, x.shape[0]):
            y_hat = sigmoid(w, x[i])
            y_diff = y_hat - y[i].item(0)
            gradient_change = y_diff * x[i]
            if reg_strength is not None:
                summed_squared = np.square(w)
                gradient_change += summed_squared * 0.5 * reg_strength
            gradient = np.add(gradient, gradient_change)

        result = learning_rate * gradient

        w = np.subtract(w, result)

        training_accuracies.append(check_accuracy(w, x, y))
        testing_accuracies.append(check_accuracy(w, x1, y1))

    return training_accuracies, testing_accuracies


args = sys.argv[1:]
if len(args) < 3:
    print("You must include a training and testing dataset, as well as a learning rate", file=sys.stderr)
    print(
        "Like so: python3 q2_3.py training_data testing_data [list of regularization strengths]")
    exit(1)

trx, tre = loadCSV(args[0], 256)
tex, tee = loadCSV(args[1], 256)

iterations = [x for x in range(1, 101)]
for i in args[2:]:
    print(f"Testing with {i} regularization")
    training_accuracies, testing_accuracies = gradient(trx, tre, tex, tee, i)
    plt.ylabel("Accuracy")
    plt.xlabel("Iteration")
    plt.title(
        f"Accuracy as Function of Iteration {f'With Lambda = {i}' if i is not None else ''}")
    plt.plot(iterations, training_accuracies, 'b', label='training')
    plt.plot(iterations, testing_accuracies, 'r', label='testing')
    plt.legend()
    # plt.show()
    plt.savefig(f"graph_results_reg_{i}.png")
    plt.clf()
