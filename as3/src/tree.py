import numpy as np
import random


class Node():
    """
    Node of decision tree

    Parameters:
    -----------
    prediction: int
            Class prediction at this node
    feature: int
            Index of feature used for splitting on
    split: int
            Categorical value for the threshold to split on for the feature
    left_tree: Node
            Left subtree
    right_tree: Node
            Right subtree
    """

    def __init__(self, prediction, feature, split, left_tree, right_tree):
        self.prediction = prediction
        self.feature = feature
        self.split = split
        self.left_tree = left_tree
        self.right_tree = right_tree


class DecisionTreeClassifier():
    """
    Decision Tree Classifier. Class for building the decision tree and making predictions

    Parameters:
    ------------
    max_depth: int
            The maximum depth to build the tree. Root is at depth 0, a single split makes depth 1 (decision stump)
    """

    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    # take in features X and labels y
    # build a tree
    def fit(self, X, y):
        self.num_classes = len(set(y))
        self.root = self.build_tree(X, y, depth=1)

    # make prediction for each example of features X
    def predict(self, X):
        preds = [self._predict(example) for example in X]

        return preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):
        node = self.root
        while node.left_tree:
            if example[node.feature] < node.split:
                node = node.left_tree
            else:
                node = node.right_tree
        return node.prediction

    # accuracy
    def accuracy_score(self, X, y):
        preds = self.predict(X)
        accuracy = (preds == y).sum() / len(y)
        return accuracy

    # function to build a decision tree
    def build_tree(self, X, y, depth, features=0):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on

        self.features_idx = np.arange(0, X.shape[1])

        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_split = None
        best_gain = 0.0
        best_left_X = None
        best_left_y = None
        best_right_X = None
        best_right_y = None

        # what we would predict at this node if we had to
        # majority class
        num_samples_per_class = [np.sum(y == i)
                                 for i in range(self.num_classes)]
        prediction = np.argmax(num_samples_per_class)
        # consider each feature
        if features != 0:
            # generate list of max_features
            kept_features = set([])
            while len(kept_features) < features:
                # random number between 0 and 50
                kept_features.add(random.randint(0, X.shape[1] - 1))
            self.features_idx = kept_features

        for feature in self.features_idx:
            # consider the set of all values for that feature to split on
            possible_splits = np.unique(X[:, feature])
            for split in possible_splits:
                # get the gain and the data on each side of the split
                # >= split goes on right, < goes on left
                gain, left_X, right_X, left_y, right_y = self.check_split(
                    X, y, feature, split)
                # if we have a better gain, use this split and keep track of data
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_split = split
                    best_left_X = left_X
                    best_right_X = right_X
                    best_left_y = left_y
                    best_right_y = right_y

        # if we haven't hit a leaf node
        # add subtrees recursively
        if best_gain > 0.0:
            left_tree = self.build_tree(
                best_left_X, best_left_y, depth=depth+1)
            right_tree = self.build_tree(
                best_right_X, best_right_y, depth=depth+1)
            return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)

    # gets data corresponding to a split by using numpy indexing

    def check_split(self, X, y, feature, split):
        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]

        # calculate gini impurity and gain for y, left_y, right_y
        gain = self.calculate_gini_gain(y, left_y, right_y)
        return gain, left_X, right_X, left_y, right_y

    def calculate_gini_gain(self, y, left_y, right_y):
        # not a leaf node
        # calculate gini impurity and gain
        gain = 0

        if len(left_y) > 0 and len(right_y) > 0:

            ########################################
            #       YOUR CODE GOES HERE            #
            ########################################
            c_plus = 0
            c_minus = 0
            cl_plus = 0
            cl_minus = 0
            cr_plus = 0
            cr_minus = 0
            pl = 0
            pr = 0

            for i in range(0, len(y)):
                if y[i] == 1:
                    c_plus += 1
                else:
                    c_minus += 1
            t1 = (c_plus / (c_plus + c_minus))
            t2 = (c_minus / (c_plus + c_minus))
            u_y = 1 - (t1 ** 2) - (t2 ** 2)
            for j in range(0, len(left_y)):
                if left_y[j] == 1:
                    cl_plus += 1
                else:
                    cl_minus += 1
            t1 = (cl_plus / (cl_plus + cl_minus))
            t2 = (cl_minus / (cl_plus + cl_minus))
            u_left_y = 1 - (t1 ** 2) - (t2 ** 2)
            for k in range(0, len(right_y)):
                if right_y[k] == 1:
                    cr_plus += 1
                else:
                    cr_minus += 1
            t1 = (cr_plus / (cr_plus + cr_minus))
            t2 = (cr_minus / (cr_plus + cr_minus))
            u_right_y = 1 - (t1 ** 2) - (t2 ** 2)
            pl = (cl_plus + cl_minus) / (c_plus + c_minus)
            pr = (cr_plus + cr_minus) / (c_plus + c_minus)
            gain = u_y - (pl * u_left_y) - (pr * u_right_y)
            return gain
        # we hit leaf node
        # don't have any gain, and don't want to divide by 0
        else:
            return 0


class RandomForestClassifier():
    """
    Random Forest Classifier. Build a forest of decision trees.
    Use this forest for ensemble predictions

    YOU WILL NEED TO MODIFY THE DECISION TREE VERY SLIGHTLY TO HANDLE FEATURE BAGGING

    Parameters:
    -----------
    n_trees: int
            Number of trees in forest/ensemble
    max_features: int
            Maximum number of features to consider for a split when feature bagging
    max_depth: int
            Maximum depth of any decision tree in forest/ensemble
    """

    def __init__(self, n_trees, max_features, max_depth):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth

        ##################
        # YOUR CODE HERE #
        ##################
        # make n trees

    # fit all trees

    def fit(self, X, y):
        bagged_X, bagged_y = self.bag_data(X, y)
        print('Fitting Random Forest...\n')
        trees = []
        for i in range(self.n_trees):
            print(i+1, end='\t\r')
            ##################
            # YOUR CODE HERE #
            ##################
            x = DecisionTreeClassifier(max_depth=self.max_depth)
            x.num_classes = len(set(bagged_y[i]))
            x.root = x.build_tree(
                bagged_X[i], bagged_y[i], depth=1, features=self.max_features)
            trees.append(x)
            preds_train = x.predict(bagged_X[i])

            train_accuracy = x.accuracy_score([preds_train], bagged_y[i])

            print(f"Train {i} accuracy {train_accuracy}")

            ##################
        print()

    def bag_data(self, X, y, proportion=1.0):
        bagged_X = []
        bagged_y = []
        for i in range(self.n_trees):
            ##################
            # YOUR CODE HERE #
            ##################
            temp_X = []
            temp_y = []
            for j in range(0, 2097):
                idx = random.randint(0, 2097)
                temp_X.append(X[idx])
                temp_y.append(y[idx])
            bagged_X.append(temp_X)
            bagged_y.append(temp_y)
        ##################
        # ensure data is still numpy arrays
        return np.array(bagged_X), np.array(bagged_y)

    def predict(self, X):
        preds = []

        # remove this one \/
        # preds = np.ones(len(X)).astype(int)
        # ^that line is only here so the code runs

        ##################
        # YOUR CODE HERE #
        ##################
        f = []
        for i in range(self.max_features):
            f.append(random.choice(X))
        preds = [self._predict(example) for example in f]
        ##################
        return preds

    def _predict(self, example):
        # node = self.root
        while node.left_tree:
            if example[node.feature] < node.split:
                node = node.left_tree
            else:
                node = node.right_tree
        return node.prediction

################################################
# YOUR CODE GOES IN ADABOOSTCLASSIFIER         #
# MUST MODIFY THIS EXISTING DECISION TREE CODE #
################################################


class AdaBoostClassifier():
    def __init__(self, trees):
        self.trees = trees

    def fit(self, x_train, y_train):
        y_train[y_train == 0] = -1
        self.x_train = x_train
        self.y_train = y_train
        self.ada_boost(x_train, y_train)

    def calculate_alpha(self, e_t):
        term = (1 - e_t) / e_t
        alpha = (1/2) * np.log(term)
        return alpha

    def ada_boost(self, X, y):
        self.errors = [0] * self.trees
        self.alphas = [0] * self.trees
        self.weights = [[0]] * self.trees
        self.classifier = [0] * self.trees
        for t in range(self.trees):
            if t == 0:
                self.weights[0] = [1 / (X.shape[0])] * X.shape[0]
                self.alphas[0] = 1

            self.classifier[t] = DecisionTreeAda(
                max_depth=1, weights=self.weights[t if t == 0 else t - 1], alpha=self.alphas[t if t == 0 else t - 1])
            self.classifier[t].fit(X, y)
            # calculate error
            self.errors[t] = self.classifier[t].get_weighted_error(X, y)
            print(f"Error[{t}]: {self.errors[t]}")
            # calculate alpha
            self.alphas[t] = self.calculate_alpha(self.errors[t])
            print(f"Alpha[{t}]: {self.alphas[t]}")
            self.weights[t] = self.classifier[t].get_modified_weights(
                X, y, self.alphas[t])

    def predict(self, X):
        best_accuracy_score = 0
        classifier = None
        best = 0
        for i in range(self.trees):
            score = self.classifier[i].accuracy_score(
                self.x_train, self.y_train)
            if best_accuracy_score < score:
                best_accuracy_score = score
                classifier = self.classifier[i]
                best = i

        print(f"Best classifier was {i}")
        prediction = np.array(classifier.predict(X))
        return prediction


class DecisionTreeAda():
    # Weights is D
    def __init__(self, weights=None, alpha=0, max_depth=1):
        self.weights = weights
        self.alpha = alpha
        self.preds = None
        self.max_depth = max_depth

    # take in features X and labels y
    # build a tree
    def fit(self, X, y):
        self.num_classes = len(set(y))
        self.root = self.build_tree(X, y, depth=1)

    # make prediction for each example of features X
    def predict(self, X):
        self.preds = [self._predict(example) for example in X]
        return self.preds

    # prediction for a given example
    # traverse tree by following splits at nodes
    def _predict(self, example):
        node = self.root
        while node.left_tree:
            if example[node.feature] < node.split:
                node = node.left_tree
            else:
                node = node.right_tree
        return node.prediction

    # accuracy
    def accuracy_score(self, X, y):
        preds = self.predict(X)
        accuracy = (preds == y).sum() / len(y)
        return accuracy

    def get_weighted_error(self, X, y):
        preds = self.predict(X)
        weight = 0
        for i in range(len(preds)):
            if preds[i] != y[i]:
                weight += self.weights[i]
        return weight

    def get_modified_weights(self, X, y, alpha):
        preds = self.predict(X)
        weights = self.weights
        for i in range(len(weights)):
            if preds[i] == y[i]:
                weights[i] = weights[i] * (-np.e ** alpha)
            else:
                weights[i] = weights[i] * (np.e ** alpha)
        return weights

    # function to build a decision tree

    def build_tree(self, X, y, depth, features=0):
        num_samples, num_features = X.shape
        # which features we are considering for splitting on

        self.features_idx = np.arange(0, X.shape[1])

        # store data and information about best split
        # used when building subtrees recursively
        best_feature = None
        best_split = None
        best_gain = 0.0
        best_left_X = None
        best_left_y = None
        best_right_X = None
        best_right_y = None

        # what we would predict at this node if we had to
        # majority class
        num_samples_per_class = [np.sum(y == i)
                                 for i in range(self.num_classes)]
        prediction = np.argmax(num_samples_per_class)

        # if we haven't hit the maximum depth, keep building
        if depth <= self.max_depth:
            # consider each feature
            if features != 0:
                # generate list of max_features
                kept_features = set([])
                while len(kept_features) < features:
                    # random number between 0 and 50
                    kept_features.add(random.randint(0, X.shape[1] - 1))
                self.features_idx = kept_features

            for feature in self.features_idx:
                # consider the set of all values for that feature to split on
                possible_splits = np.unique(X[:, feature])
                for split in possible_splits:
                    # get the gain and the data on each side of the split
                    # >= split goes on right, < goes on left
                    gain, left_X, right_X, left_y, right_y = self.check_split(
                        X, y, feature, split)
                    # if we have a better gain, use this split and keep track of data
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_split = split
                        best_left_X = left_X
                        best_right_X = right_X
                        best_left_y = left_y
                        best_right_y = right_y

        # if we haven't hit a leaf node
        # add subtrees recursively
        if best_gain > 0.0:
            left_tree = self.build_tree(
                best_left_X, best_left_y, depth=depth+1)
            right_tree = self.build_tree(
                best_right_X, best_right_y, depth=depth+1)
            return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=left_tree, right_tree=right_tree)

        # if we did hit a leaf node
        return Node(prediction=prediction, feature=best_feature, split=best_split, left_tree=None, right_tree=None)

    # gets data corresponding to a split by using numpy indexing

    def check_split(self, X, y, feature, split):
        left_idx = np.where(X[:, feature] < split)
        right_idx = np.where(X[:, feature] >= split)
        left_X = X[left_idx]
        right_X = X[right_idx]
        left_y = y[left_idx]
        right_y = y[right_idx]

        # calculate gini impurity and gain for y, left_y, right_y
        gain = self.calculate_gini_gain(y, left_y, right_y)
        return gain, left_X, right_X, left_y, right_y

    def calculate_gini_gain(self, y, left_y, right_y):
        # not a leaf node
        # calculate gini impurity and gain
        gain = 0

        if len(left_y) > 0 and len(right_y) > 0:

            ########################################
            #       YOUR CODE GOES HERE            #
            ########################################
            c_plus = 0
            c_minus = 0
            cl_plus = 0
            cl_minus = 0
            cr_plus = 0
            cr_minus = 0
            pl = 0
            pr = 0

            for i in range(0, len(y)):
                if y[i] == 1:
                    c_plus += 1
                else:
                    c_minus += 1

            t1 = (c_plus / (c_plus + c_minus))
            t2 = (c_minus / (c_plus + c_minus))
            u_y = 1 - (t1 ** 2) - (t2 ** 2)

            for j in range(0, len(left_y)):
                if left_y[j] == 1:
                    cl_plus += 1
                else:
                    cl_minus += 1

            t1 = (cl_plus / (cl_plus + cl_minus))
            t2 = (cl_minus / (cl_plus + cl_minus))
            u_left_y = 1 - (t1 ** 2) - (t2 ** 2)

            for k in range(0, len(right_y)):
                if right_y[k] == 1:
                    cr_plus += 1
                else:
                    cr_minus += 1

            t1 = (cr_plus / (cr_plus + cr_minus))
            t2 = (cr_minus / (cr_plus + cr_minus))
            u_right_y = 1 - (t1 ** 2) - (t2 ** 2)
            pl = (cl_plus + cl_minus) / (c_plus + c_minus)
            pr = (cr_plus + cr_minus) / (c_plus + c_minus)
            gain = u_y - (pl * u_left_y) - (pr * u_right_y)

            return gain
        # we hit leaf node
        # don't have any gain, and don't want to divide by 0
        else:
            return 0
