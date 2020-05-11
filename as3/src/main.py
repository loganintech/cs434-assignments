from tree import DecisionTreeClassifier, RandomForestClassifier, AdaBoostClassifier
from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def load_args():

    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--county_dict', default=1, type=int)
    parser.add_argument('--decision_tree', default=1, type=int)
    parser.add_argument('--random_forest', default=1, type=int)
    parser.add_argument('--ada_boost', default=1, type=int)
    parser.add_argument('--root_dir', default='../data/', type=str)
    args = parser.parse_args()

    return args


def county_info(args):
    county_dict = load_dictionary(args.root_dir)
    dictionary_info(county_dict)


def decision_tree_testing(x_train, y_train, x_test, y_test):
    print('Decision Tree\n\n')
    clf = DecisionTreeClassifier(max_depth=25)
    clf.fit(x_train, y_train)
    preds_train = clf.predict(x_train)
    preds_test = clf.predict(x_test)
    train_accuracy = accuracy_score(preds_train, y_train)
    test_accuracy = accuracy_score(preds_test, y_test)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = clf.predict(x_test)
    print('F1 Test {}'.format(f1(y_test, preds)))


def random_forest_testing(x_train, y_train, x_test, y_test):
    print('Random Forest\n\n')
    rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=10)
    rclf.fit(x_train, y_train)
    preds_train = rclf.predict(x_train)
    preds_test = rclf.predict(x_test)
    train_accuracy = accuracy_score(preds_train, y_train)
    test_accuracy = accuracy_score(preds_test, y_test)
    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = rclf.predict(x_test)
    print('F1 Test {}'.format(f1(y_test, preds)))


def ada_boost_testing(x_train, y_train, x_test, y_test, trees):
    print('AdaBoost\n\n')
    ada = AdaBoostClassifier(trees=trees)

    ada.fit(x_train, y_train)
    preds_train = ada.predict(x_train)
    preds_test = ada.predict(x_test)
    train_accuracy = accuracy_score(preds_train, y_train)
    test_accuracy = accuracy_score(preds_test, y_test)

    print('Train {}'.format(train_accuracy))
    print('Test {}'.format(test_accuracy))
    preds = ada.predict(x_test)
    print('F1 Test {}'.format(f1(y_test, preds)))


def create_trees(x_train, y_train, x_test, y_test):
    train = []
    test = []
    y = []
    for i in range(1, 26):
        y.append(i)
        print("LENGTH: ", i)
        h = DecisionTreeClassifier(max_depth=i)
        h.fit(x_train, y_train)
        preds_train = h.predict(x_train)
        preds_test = h.predict(x_test)
        train_accuracy = accuracy_score(preds_train, y_train)
        train.append(train_accuracy)
        test_accuracy = accuracy_score(preds_test, y_test)
        test.append(test_accuracy)
        print('Train {}'.format(train_accuracy))
        print('Test {}'.format(test_accuracy))
        preds = h.predict(x_test)
        print('F1 Test {}'.format(f1(y_test, preds)))
    return y, train, test


###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
    args = load_args()
    x_train, y_train, x_test, y_test = load_data(args.root_dir)
    if args.county_dict == 1:
        county_info(args)
    if args.decision_tree == 1:
        decision_tree_testing(x_train, y_train, x_test, y_test)
    if args.random_forest == 1:
        random_forest_testing(x_train, y_train, x_test, y_test)
    if args.ada_boost == 1:
        for i in range(30, 200, 10):
            print(f"Testing AdaBoost with {i} trees")
            ada_boost_testing(x_train, y_train, x_test, y_test, i)

    print('Done')
