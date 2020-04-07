import numpy as np
import sys


def run_tests(training, testing, should_add_dummy=True):

    print(f"Running tests {'with' if should_add_dummy else 'without'} dummy column.")

    # Load our training data
    (_, cols) = training.shape
    expected = training[:, -1]
    modified_features = np.delete(training, cols-1, 1)

    if should_add_dummy:
        modified_features = np.insert(modified_features, 0, 1, axis=1)
    (_, cols) = modified_features.shape

    # Features = X so f_transposed = X^T
    f_transposed = np.transpose(modified_features)
    # Combine X and X^T to get first term
    first_term = np.matmul(f_transposed, modified_features)
    # And then raise that term to the -1 (inverting the matrix)
    inverted = np.linalg.inv(first_term)
    # Then we calculate our second term (X^T * Y)
    second_term = np.matmul(f_transposed, expected)
    # And multiply the terms together
    weight = np.matmul(inverted, second_term)
    weight_transposed = np.transpose(weight.reshape(cols, 1))
    # print(weight)

    # print(f"Learned weight vector:\n{weight}\n")

    test_expected = testing[:, -1]

    (_, test_cols) = testing.shape
    modified_test = np.delete(testing, test_cols-1, 1)
    if should_add_dummy:
        modified_test = np.insert(modified_test, 0, 1, axis=1)

    test_transposed = np.transpose(modified_test)

    train_estimates = np.matmul(weight, np.transpose(modified_features))
    # print(train_estimates)
    test_estimates = np.matmul(weight, test_transposed)
    # print(test_estimates)

    ase_train = sum([(real - estimate) ** 2 for (real, estimate) in zip(expected, train_estimates)]) / len(train_estimates)
    ase_test = sum([(real - estimate) ** 2 for (real, estimate) in zip(test_expected, test_estimates)]) / len(test_estimates)

    # print(f"Average Squared Error for Training Data: {ase_train}")
    # print(f"Average Squared Error for Test Data: {ase_test}")
    return {
        "weight": weight,
        "ase_train": ase_train,
        "ase_test": ase_test,
    }

def print_results(results):
    print(f"Learned weight vector:\n{results['weight']}\n")
    print(f"Average Squared Error for Training Data: {results['ase_train']}")
    print(f"Average Squared Error for Test Data: {results['ase_test']}")
    print()

def load_data(training, testing):
    return (np.genfromtxt(training, delimiter=","), np.genfromtxt(testing, delimiter=","))
