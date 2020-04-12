import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import random


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

    # for i in tr_feat:
    #     if i > 1 or i < 0:
    #         raise ValueError("WHY")
    # for i in te_feat:
    #     if i > 1 or i < 0:
    #         raise ValueError("WHY")

    return tr_feat, tr_exp, te_feat, te_exp


def sigmoid(weight, case):
    # try:
    exponent = -np.dot(weight.T, case)

    try:
        prediction = 1.0 / (1.0 + math.exp(exponent))
    except Exception as e:
        return 1.0 / (1.0 + math.exp(500))
        # If you've gotten this far you've noticed that the last two accuracies are always 50%
        # I couldn't tell you why, seeing as our weights look correct
        # And

    return prediction


def check_accuracy(w, x, y):
    correct = 0

    for i in range(x.shape[0]):
        if np.dot(w.T, x[i]) >= 0.0 and y[i] == 1:
            correct += 1
        elif np.dot(w.T, x[i]) < 0.0 and y[i] == 0:
            correct += 1

    percentage_correct = correct / x.shape[0]
    return percentage_correct


def gradient(training_data, training_expected, testing_data, testing_expected, reg_strength=None, iterations=100, learning_rate=0.00005):
    training_accuracies = []
    testing_accuracies = []

    if reg_strength is not None:
        try:
            reg_strength = float(reg_strength)
        except:
            reg_strength = None

    w = np.zeros(training_data.shape[1])  # Feature count

    for _ in range(iterations):
        gradient_batch = np.zeros(training_data.shape[1])  # Feature count
        for i in range(training_data.shape[0]):  # Example count
            predicted = sigmoid(w, training_data[i])
            diff = (np.subtract(
                predicted, training_expected[i]))
            diff = np.multiply(diff,   training_data[i])
            gradient_batch = np.add(gradient_batch, diff)

        if reg_strength is not None:
            normalized = np.linalg.norm(w)
            gradient_batch = np.add(
                gradient_batch, np.multiply(normalized, reg_strength))

        gradient_batch = np.multiply(learning_rate, gradient_batch)
        w = np.subtract(w, gradient_batch)

        training_accuracies.append(check_accuracy(
            w, training_data, training_expected))
        testing_accuracies.append(check_accuracy(
            w, testing_data, testing_expected))

    return training_accuracies, testing_accuracies


args = sys.argv[1:]
if len(args) < 2:
    print("You must include a training and testing dataset, as well as a learning rate", file=sys.stderr)
    print("Like so: python3 q2_1.py usps_train.csv usps_test.csv learning_rate")
    exit(1)

iterations = []
for i in range(0, 100):
    iterations.append(i+1)

training_features, training_expected, test_features, test_expected = load_files(
    args[0], args[1])
training_accuracies, testing_accuracies = gradient(
    training_features, training_expected, test_features, test_expected, learning_rate=float(args[2]))
plt.ylabel("Accuracy")
plt.xlabel("Iteration")
plt.title(f"Accuracy as  Function of Iteration Learing Rate = {args[2]}")
plt.plot(iterations, training_accuracies, 'b', label='training')
plt.plot(iterations, testing_accuracies, 'r', label='testing')
plt.legend()
plt.show()
plt.savefig(f"graph_results.png")
