from linear_regression import *
import sys

args = sys.argv[1:]
if len(args) < 2:
    print("You must include a training and testing dataset.", file=sys.stderr)
    print("Like so: python3 linear_regression.py housing_train.csv housing_test.csv")
    exit(1)

(training, testing) = load_data(args[0], args[1])
res = run_tests(training, testing)
print_results(res)
