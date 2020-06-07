from sklearn.feature_extraction.text import CountVectorizer
import re
import csv
import math
import matplotlib.pyplot as plt
import numpy as np


with open('./data/IMDB.csv', "r", encoding="utf-8") as f:

    lines = f.readlines()
    without_header = list(map(lambda line: line.split(",")[0], lines[1:]))
    training_data = without_header[:30000]
    validation_data = without_header[30000:40000]
    test_data = without_header[40000:]
    assert len(training_data) == 30000
    assert len(validation_data) == 10000
    assert len(test_data) == 10000

with open('data/IMDB_labels.csv', "r", encoding='utf-8') as f:
    lines = f.readlines()
    lines = list(map(lambda line: line.split(",")
                     [0] == "positive\n", lines[1:]))
    training_labels = lines[:30000]
    validation_labels = lines[30000:]
    assert len(training_labels) == 30000
    assert len(validation_labels) == 10000
    assert len(lines[40000:]) == 0


def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]a
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    # pattern = r'[^a-zA-z0-9\s]'
    # text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text


def learn_classes(labels):

    # Track our positive count
    positive_count = 0
    # for all the training data
    for label in labels:
        # if the label is positive, add one to our count
        if label:
            positive_count += 1

    # Pos prob is total number of pos over total count of items
    pos_prob = positive_count / len(labels)

    # learn P(y=0):
    negative_count = len(labels) - positive_count
    # neg prob is negative count / total count
    neg_prob = negative_count / len(labels)

    return pos_prob, neg_prob, positive_count, negative_count


def run_model(max_features=2000, min_df=0, max_df=1.0, alphas=[1.4], is_test_set=False, filename=None):

    # this vectorizer will skip stop words
    vectorizer = CountVectorizer(
        stop_words="english",
        preprocessor=clean_text,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )

    # fit the vectorizer on the text
    counts = vectorizer.fit(training_data)

    # # get the vocabulary
    vocab_dict = {k: v for k, v in vectorizer.vocabulary_.items()}
    count_vocab_dict = {v: k for k, v in vectorizer.vocabulary_.items()}
    vocabulary = [count_vocab_dict[i] for i in range(len(count_vocab_dict))]

    def get_vectors_from_data(data):
        vectors = []
        for line in data:
            line = clean_text(line).split()
            vec = [0] * len(vocabulary)
            for word in line:
                try:
                    vec[vocab_dict[word]] += 1
                except KeyError as e:
                    pass

            vectors.append(vec)
        return vectors

    training_vectors = get_vectors_from_data(training_data)
    validation_vectors = get_vectors_from_data(validation_data)
    test_vectors = get_vectors_from_data(test_data)
    assert len(training_vectors) == 30000
    assert len(validation_vectors) == 10000
    assert len(test_vectors) == 10000
    assert validation_vectors != test_vectors

    def train_model(is_positive=True, alpha=1):
        total_word_count = 0
        probs = [0] * len(vocabulary)  # [0, 0, 0, ...]
        # Loop over the reviews
        for rev_idx, review in enumerate(training_vectors):
            # If the review is positive, and we're not training positive, or if the review is negative, and we're not training negative, continue
            if training_labels[rev_idx] != is_positive:
                continue

            # review[i] == number of word off vocabulary[i] in the review
            # IE, if vocabulary[i] == "cheese" then review[i] is number of times "cheese" is in the review
            # Loop over the count of the i'th vocab word in said review
            for i, count_of_ith_vocab_word_in_review in enumerate(review):
                probs[i] += count_of_ith_vocab_word_in_review
                total_word_count += count_of_ith_vocab_word_in_review

        for i in range(len(probs)):
            probs[i] += alpha
            probs[i] /= total_word_count + (len(vocabulary) * alpha)
            if probs[i] != 0:
                probs[i] = math.log(probs[i])

        return probs

    def get_apply_func(alpha):
        pos_probs = train_model(is_positive=True,  alpha=alpha)
        neg_probs = train_model(is_positive=False, alpha=alpha)
        total_pos_prob, total_neg_prob, _, _ = learn_classes(training_labels)

        def apply_model(x, is_positive=True):
            positive_term = 0
            negative_term = 0

            for i in range(len(vocabulary)):
                try:
                    positive_term += pos_probs[i] * x[i]
                    negative_term += neg_probs[i] * x[i]
                except TypeError:
                    print(pos_probs[i])
                    print(x[i])

            if is_positive:
                return positive_term + math.log(total_pos_prob)
            else:
                return negative_term + math.log(total_neg_prob)

        return apply_model

    def validate(vec, labels, apply_model):
        _, _, pos, neg = learn_classes(labels)
        pos_corr = 0
        neg_corr = 0
        correct = 0
        for i, (x, posval) in enumerate(zip(vec, labels)):
            pos_prob = apply_model(x, is_positive=True)
            neg_prob = apply_model(x, is_positive=False)
            # print(pos_prob, neg_prob, pos_prob > neg_prob, posval)
            if pos_prob >= neg_prob and posval:
                pos_corr += 1
                correct += 1
            elif pos_prob < neg_prob and not posval:
                neg_corr += 1
                correct += 1
        # print("Total cor", correct / len(vec))
        # print("Pos cor", pos_corr / pos)
        # print("Neg cor", neg_corr / neg)
        return correct / len(vec)

    def write_prediction_to_csv(filename, vec, apply_model):
        with open(filename, "w") as f:
            for x in vec:
                pos_prob = apply_model(x, is_positive=True)
                neg_prob = apply_model(x, is_positive=False)
                if pos_prob >= neg_prob:
                    f.write(f"1\n")
                elif pos_prob < neg_prob:
                    f.write(f"0\n")

    if not is_test_set:
        best_res = 0
        best_alpha = -1
        res = [0] * len(alphas)
        for i, alpha in enumerate(alphas):
            f = get_apply_func(alpha)
            res[i] = validate(validation_vectors, validation_labels, f)
        return res
    elif is_test_set and filename is not None:
        f = get_apply_func(alphas[0])
        write_prediction_to_csv(filename, test_vectors, f)
    else:
        print("You must specify a filename for a test set.")


def get_best_values():
    # feature_counts = [100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000]
    feature_counts = list(map(lambda x: x * 10000, range(3, 12, 2))) # [30k, 50k, 70k, 90k, 11k]
    min_dfs = np.arange(0, 0.3, step=0.05)  # [0.0, 0.05, 0.10, ...]
    max_dfs = np.arange(1.0, 0.6, step=-0.05)  # [1.0, 0.95, 0.90, ...]

    best_feature = 0
    best_min = min(min_dfs) - 1
    best_max = max(max_dfs) + 1
    best_res = 0
    for feat in feature_counts:
        for min_df in min_dfs:
            for max_df in max_dfs:
                print(
                    f"Running model:\tFeatures: {feat}\tMin DF: {min_df}\tMax DF: {max_df}")
                try:
                    res = run_model(max_features=feat,
                                    min_df=min_df, max_df=max_df)
                except ValueError as e:
                    print(
                        f"Error running model:\n\tFeatures: {feat}\n\tMin DF: {min_df}\n\tMax DF: {max_df}")
                    continue

                if res[0] > best_res:
                    print(
                        f"Found new best model!\n\tFeatures: {feat}\n\tMin DF: {min_df}\n\tMax DF: {max_df}")
                    best_res = res[0]
                    best_min = min_df
                    best_max = max_df
                    best_feature = feat
                    print(f"Accuracy:\t{res[0]}")

    print(f"Best Accuracy: {best_res} w/ {best_feature} features")


if __name__ == "__main__":

    # # Warning this function is stupidly slow
    # get_best_values()

    run_model(is_test_set=True, filename="test-prediction1.csv", alphas=[1.0])
    run_model(is_test_set=True, filename="test-prediction2.csv", alphas=[1.4])
    run_model(max_features=50000, is_test_set=True,
              filename="test-prediction3.csv", alphas=[1.4])
