import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
import random

train = pd.read_csv("./data/train.csv", skiprows=1)
train_ids = train.iloc[:, 0].values.astype("U")
train_sentences = train.iloc[:, 1].values.astype("U")
train_phrases = train.iloc[:, 2].values.astype("U")
train_sentiments = train.iloc[:, 3].values.astype("U")

test = pd.read_csv("./data/test.csv", skiprows=1)
test_sentences = test.iloc[:, 1].values.astype("U")
test_sentiments = test.iloc[:, 2].values.astype("U")

print("Building Pipeline")
pipe = Pipeline([
    ('vect', CountVectorizer(max_df=0.8)),
    ('tfidf', TfidfTransformer(use_idf=True)),
    ('svc', SVC(C=1.2, kernel="rbf"))
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
        'vect__max_df': (0.8, 1.0),
        'vect__ngram_range': [(1, 1), (1, 2)],
        'tfidf__use_idf': (True, False),
        'svc__C': (0.8, 1.2),
        'svc__kernel': ["linear", "poly", "rbf", "sigmoid"],
    }

    gs = GridSearchCV(pipe, parameters, cv=5, n_jobs=9)
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


def randomly_extract_possible_phrases_from_sentence(sentence) -> list:
    words = sentence.split(" ")
    word_count = len(words)
    options = []
    for _ in range(20):
        option = []
        for _ in range(min(5, word_count)):
            choice = random.choice(words)
            while choice in option:
                choice = random.choice(words)
            option.append(choice)
        if option not in options:
            options.append(option)

    return [" ".join(option) for option in options]


if __name__ == "__main__":
    # simple_fitting(pipe)
    combined = combined_fitting(pipe)
    dump(combined, 'combined_pipe.joblib')
    for sentence in test_sentences[:10]:
        print(sentence)
        phrases = randomly_extract_possible_phrases_from_sentence(sentence)
        print(phrases)
        probs = []
        for phrase in phrases:
            probs.append(combined.predict_proba(phrase))

        idx = probs.index(max(probs))
        print(phrases[idx])
