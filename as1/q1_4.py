from linear_regression import *
import sys
from numpy.random import standard_normal
import numpy as np
import matplotlib.pyplot as plt

args = sys.argv[1:]
if len(args) < 2:
    print("You must include a training and testing dataset.", file=sys.stderr)
    print("Like so: python3 linear_regression.py housing_train.csv housing_test.csv")
    exit(1)


(training, testing) = load_data(args[0], args[1])
(tr_row, tr_col) = training.shape
(te_row, te_col) = testing.shape

ase_train_res = []
ase_test_res = []

range_arr = range(2, 2000, 2)

for d in range_arr:
    train_dist = standard_normal(d * tr_row).reshape(d, tr_row)
    longer_training = np.insert(training, 0, train_dist, axis=1)
    test_dist = standard_normal(d * te_row).reshape(d, te_row)
    longer_testing = np.insert(testing, 0, test_dist, axis=1)

    print(f"Running tests with {d} new columns.")
    res = run_tests(longer_training, longer_testing)
    ase_train_res.append(res['ase_train'])
    ase_test_res.append(res['ase_test'])

plt.ylabel("Mean Squared Error")
plt.xlabel("Random Features 'D'")
plt.title("Training vs Test Error over Random Features D")
training, = plt.plot(range_arr, ase_train_res, label="Training Data")
testing, = plt.plot(range_arr, ase_test_res, label="Testing Data")
plt.legend(handles=[training, testing])
plt.savefig(f"linreg_results_{len(range_arr)*2}.png")
