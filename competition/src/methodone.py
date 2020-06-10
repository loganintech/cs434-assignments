import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import random
import re
import os

# Will be true when running in Kaggle, false otherwise
use_kaggle_filepaths = os.environ.get("KAGGLE_DATA_PROXY_TOKEN") is not None
use_svm = False
load_from_joblib = False

if use_kaggle_filepaths:
    train_filepath = "/kaggle/input/tweet-sentiment-extraction/train.csv"
    test_filepath = "/kaggle/input/tweet-sentiment-extraction/test.csv"
    output_filepath = "/kaggle/working/submission.csv"
    svm_joblib_filepath = "/kaggle/input/svcprobabiltiytrainedjoblibtweetextraction/v0.0.2p.joblib"
    neural_joblib_filepath = "/kaggle/input/svcprobabiltiytrainedjoblibtweetextraction/v0.0.2p.joblib"
else:
    train_filepath = "./data/train.csv"
    test_filepath = "./data/test.csv"
    output_filepath = "./data/output.csv"
    svm_joblib_filepath = "./svm/v0.0.2p.joblib"
    neural_joblib_filepath = "./neural/v0.0.0m.joblib"

word_regex = re.compile(r"[\w ]+")

train = pd.read_csv(train_filepath, skiprows=1)
train_ids = train.iloc[:, 0].values.astype("U")
train_sentences = train.iloc[:, 1].values.astype("U")
train_phrases = train.iloc[:, 2].values.astype("U")
train_sentiments = train.iloc[:, 3].values.astype("U")

test = pd.read_csv(test_filepath, skiprows=1)
test_ids = test.iloc[:, 0].values.astype("U")
test_sentences = test.iloc[:, 1].values.astype("U")
test_sentiments = test.iloc[:, 2].values.astype("U")

svm_pipe = Pipeline([
    ('vect', CountVectorizer(max_df=0.8, max_features=10000)),
    ('tfidf', TfidfTransformer(use_idf=True, sublinear_tf=True)),
    ('svc', SVC(C=1.2, kernel="rbf", probability=True,
                cache_size=1000, break_ties=True))
])
neural_pipe = Pipeline([
    ('vect', CountVectorizer(max_df=0.8, max_features=10000)),
    ('tfidf', TfidfTransformer(use_idf=True, sublinear_tf=True)),
    ('mlp', MLPClassifier())
])


def simple_fitting(pipe):
    print("Simple Fitting=", end="")
    pipe.fit(train_phrases, train_sentiments)
    print(pipe.score(test_sentences, test_sentiments))
    return pipe


# Disclaimer, this takes about an hour to run on my desktop
def run_param_checker():
    print("Running Paramater Checker, Come Back in an Hour")
    # First time results:
    # svc__C: 1.2
    # svc__kernel: 'rbf'
    # tfidf__use_idf: True
    # vect__max_df: 0.8
    # vect__ngram_range: (1, 1)

    parameters = {
        ##########
        # Below are the first-run parameters we used for the grid search
        # IE, the first time we tried optimizing, we used those settings
        # 'vect__max_df': (0.8, 1.0),
        # 'vect__ngram_range': [(1, 1), (1, 2)],
        # 'tfidf__use_idf': (True, False),
        # 'svc__C': (0.8, 1.2),
        # 'svc__kernel': ["linear", "poly", "rbf", "sigmoid"],
        # Second run parameters:
        # svc__break_ties: True
        # tfidf__sublinear_tf: True
        # vect__analyzer: 'word'
        # vect__lowercase: True
        # vect__max_features: 10000
        # Third run parameters
        # 'ada__n_estimators': 100,
        # 'ada__learning_rate': 0.9

        # Here lies the training set I tried that one time, that took 160 minutes and was cancelled.
        # 'mlp__hidden_layer_sizes': [(100,), (80, 100), (100, 80), (200, 200)],
        # 'mlp__activation': ["logistic", "tanh", "relu"],
        # 'mlp__solver': ["sgd", "adam"],
        # 'mlp__alpha': [0.001, 0.0001, 0.00001],
    }

    gs = GridSearchCV(pipe, parameters, cv=5, n_jobs=12)
    gs.fit(train_phrases, train_sentiments)

    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs.best_params_[param_name]))


def combined_fitting(pipe):
    print("Combined Fitting=", end="")
    combined_x = np.concatenate((train_sentences, train_phrases))
    combined_y = np.concatenate((train_sentiments, train_sentiments))
    pipe.fit(combined_x, combined_y)
    print(pipe.score(test_sentences, test_sentiments))
    return pipe


def randomly_extract_possible_phrases_from_sentence(sentence, inner_iteration_max=20) -> list:
    # print(f"Pre Regex: {sentence}")
    sentence = sentence.replace("'", '')
    sentence = sentence.replace("`", '')
    sentence = word_regex.findall(sentence)
    sentence = list(filter(lambda x: len(x) > 0 and x != " ", sentence))
    # print(f"Post 0Length Filter: {sentence}")
    for idx, part in enumerate(sentence):
        # print(f"Part: {part}")
        part = part.split(" ")
        part = list(filter(lambda x: len(x) > 0, part))
        sentence[idx] = " ".join(part)
    # print(f"Post Space Removal: {sentence}")

    sentence = list(filter(lambda x: len(x) > 1, sentence))
    sentence = " ".join([part for part in sentence])
    # print(sentence)
    words = sentence.split(" ")

    word_count = len(set(words))
    if word_count < 2:
        return [sentence]
    options = []
    for _ in range(30):
        option = []
        last_idx = 0
        for _ in range(min((int(random.random()) * 5) + 3, word_count)):
            choice = random.choice(words)
            idx = words.index(choice)
            iters = 0
            while (choice in option or idx < last_idx) and iters < inner_iteration_max:
                choice = random.choice(words)
                iters += 1
                idx = words.index(choice)

            option.append(choice)
        if option not in options:
            options.append(option)

    final = []
    for option in options:
        if len(option) > 1:
            final.append(" ".join(option))

    return final


def array_idx_from_label(label):
    return {
        "negative": 0,
        "neutral": 1,
        "positive": 2
    }[label]


def predict_from_pipe(pipe):
    with open(output_filepath, "w") as f:
        f.write("textID,selected_text\n")

        for id_idx, sentence in enumerate(test_sentences):
            # print(f"Full Sentence: {sentence}")
            phrases = randomly_extract_possible_phrases_from_sentence(sentence)
            # print(phrases)
            sentence_label = pipe.predict([sentence])[0]
            # prob_label = pipe.predict(phrases)
            probs = pipe.predict_proba(phrases)
            # for phrase in phrases:
            # probs.append(combined.predict([phrase]))

            # print(sentence_label)
            # print(probs)
            idx = 0
            highest = 0
            for i, prob in enumerate(probs):
                # for i, (prob, label) in enumerate(zip(probs, prob_label)):
                prob = prob[array_idx_from_label(sentence_label)]
                if prob > highest:
                    idx = i
                    highest = prob
            # print(f"{sentence_label} - {test_ids[idx]}: {phrases[idx]}")
            f.write(f"{test_ids[id_idx]},{phrases[idx]}\n")
            f.flush()


if __name__ == "__main__":
    # run_param_checker()

    # simple_fitting(pipe)
    if load_from_joblib:
        if use_svm:
            pipe = load(svm_joblib_filepath)
        else:
            pipe = load(neural_joblib_filepath)
    else:
        if use_svm:
            pipe = combined_fitting(svm_pipe)
        else:
            pipe = combined_fitting(neural_pipe)

    predict_from_pipe(pipe)
