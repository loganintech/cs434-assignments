import sys
import numpy as np
import math as mth
import csv
import matplotlib.pyplot as plt


def load_files(training, testing):
    tr_feat = np.genfromtxt(training, usecols=range(256), delimiter=",")
    tr_feat /= 255.0
    tr_feat = np.insert(tr_feat, 0, 0, axis=1)
    tr_exp = np.genfromtxt(training, usecols=range(-1), delimiter=",")
    tr_exp = tr_exp[:, -1]

    te_feat = np.genfromtxt(testing, usecols=range(256), delimiter=",")
    te_feat /= 255.0
    te_feat = np.insert(te_feat, 0, 0, axis=1)
    te_exp = np.genfromtxt(testing, usecols=range(-1), delimiter=",")
    te_exp = te_exp[:, -1]

    return tr_feat, tr_exp, te_feat, te_exp


def calculate_y_hat(w, x):
    exponent = (-1*np.dot(w.T, x))
    new_exponent = float(exponent.item(0))
    y_hat = 1./(1. + mth.exp(new_exponent))
    return y_hat


def check_accuracy(w, x, y):
    correct = 0

    for i in range(0, x.shape[0]):
        if y[i] == 1:
            sigmoid = calculate_y_hat(w, x[i])
            if sigmoid >= 0.5:
                correct += 1
        elif y[i] == 0:
            sigmoid_complement = 1 - calculate_y_hat(w, x[i])
            if sigmoid_complement >= 0.5:
                correct += 1

    percentage_correct = correct/x.shape[0]
    return percentage_correct


def gradient(x, y, x1, y1):
    # **************************************************************************
    # LOGISTIC REGRESSION
    # **************************************************************************
    training_accuracies = []
    testing_accuracies = []

    w = np.zeros(x.shape[1])

    learning_rate = 0.001  # let the learning rate be 0.001
    iterations = 0  # number of times it has iterated through the while loop

    while True:
        gradient = np.zeros(x.shape[1])
        for i in range(1, x.shape[0]):
            y_hat = calculate_y_hat(w, x[i])
            a = y_hat - y[i].item(0)
            b = a*x[i]
            gradient = np.add(gradient, b)

        result = learning_rate * gradient
        w = np.subtract(w, result)

        training_accuracies.append(check_accuracy(w, x, y))
        testing_accuracies.append(check_accuracy(w, x1, y1))

        iterations += 1
        if iterations == 100:
            break
    return training_accuracies, testing_accuracies


args = sys.argv[1:]
if len(args) < 2:
    print("You must include a training and testing dataset, as well as a learning rate", file=sys.stderr)
    print("Like so: python3 q2_1.py usps_train.csv usps_test.csv")
    exit(1)

iterations = []

for i in range(0, 100):
    iterations.append(i+1)

training_features, training_expected, test_features, test_expected = load_files(
    args[0], args[1])
training_accuracies, testing_accuracies = gradient(
    training_features, training_expected, test_features, test_expected)
plt.ylabel("Accuracy")
plt.xlabel("Iteration")
plt.title("Accuracy as  Function of Iteration")
plt.plot(iterations, training_accuracies, 'b', label='training')
plt.plot(iterations, testing_accuracies, 'r', label='testing')
plt.legend()
plt.show()
plt.savefig(f"graph_results.png")
