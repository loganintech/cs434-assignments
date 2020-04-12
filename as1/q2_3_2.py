import sys
import numpy as np
import math
from mpmath import mp
import csv
import matplotlib.pyplot as plt


def load_files(training, testing):
    trdata = np.genfromtxt(training, delimiter=",")
    tedata = np.genfromtxt(testing, delimiter=",")
    tr_feat = trdata[:, :-1] / 255.0
    tr_exp = trdata[:, -1]
    te_feat = tedata[:, :-1] / 255.0
    te_exp = tedata[:, -1]
    return tr_feat, tr_exp, te_feat, te_exp


def sigmoid(weight, case):
    exponent = np.dot(-weight, case)

    try:
        prediction = 1.0 / (1.0 + math.exp(exponent.item(0)))
    except Exception as e:
        # print(f"Exponent: {exponent}")
        # print(f"Exponent Item: {exponent.item(0)}")
        prediction = 1.0 / (1.0 + mp.exp(exponent.item(0)))
    return prediction


def check_accuracy(w, x, y):
    correct = 0

    for (case, expected) in zip(x, y):
        if expected == 1 and sigmoid(w, case) >= 0.5:
            correct += 1
        elif expected == 0 and 1 - sigmoid(w, case) >= 0.5:
            correct += 1

    percentage_correct = correct / x.shape[0]
    return percentage_correct


def gradient(training_data, training_expected, testing_data, testing_expected, reg_strength=None, iterations=100):
    training_accuracies = []
    testing_accuracies = []

    if reg_strength is not None:
        try:
            reg_strength = float(reg_strength)
        except:
            reg_strength = None

    w = np.zeros((1, 256))
    regularization_val = 0

    learning_rate = 0.000005  # let the learning rate be very small

    for _ in range(iterations):
        batch = np.zeros((1, 256))
        for i, case in enumerate(training_data):
            predicted = sigmoid(w, case)
            diff = (predicted - training_expected[i]) * case
            batch = np.add(batch, diff)

        if reg_strength is not None:
            normalized = np.linalg.norm(w) ** 2
            batch = np.add(batch, normalized * 0.5 * reg_strength)

        batch = learning_rate * batch
        w = np.subtract(w, batch)

    train_acc = check_accuracy(w, training_data, training_expected)
    test_acc = check_accuracy(w, testing_data, testing_expected)

    return train_acc, test_acc


if __name__ == "__main__":

    args = sys.argv[1:]
    if len(args) < 3:
        print("You must include a training and testing dataset, as well as a learning rate", file=sys.stderr)
        print(
            "Like so: python3 q2_3.py training_data testing_data [list of regularization strengths]")
        exit(1)

    training_features, training_expected, test_features, test_expected = load_files(
        args[0], args[1])

    tr_acc, te_acc = [], []
    for regularization_strength in args[2:]:
        print(f"Regularization strength {regularization_strength}")
        training_acc, testing_acc = gradient(
            training_features, training_expected, test_features, test_expected, regularization_strength)
        tr_acc.append(training_acc)
        te_acc.append(testing_acc)

    plt.ylabel("Accuracy")
    plt.xlabel("Lambda")
    plt.title(
        f"Accuracy as Function of Lambda")
    plt.plot(args[2:], tr_acc, 'b', label='training')
    plt.plot(args[2:], te_acc, 'r', label='testing')
    plt.legend()
    # plt.show()
    plt.savefig(f"logistic_regression.png")
    plt.clf()
