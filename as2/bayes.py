import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import csv
import math


def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
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


def learn_classes():
    # learn P(y=1):
    with open('data/IMDB_labels.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    data.pop(0)

    positive_count = 0
    for i in range(0, 30000):
        if data[i] == ['positive']:
            positive_count += 1

    pos_prob = positive_count/30000

    # learn P(y=0):
    negative_count = 30000 - positive_count
    neg_prob = negative_count/30000

    return pos_prob, neg_prob


# Importing the dataset
training_data = pd.read_csv('data/IMDB.csv', delimiter=',', nrows=30000)


# this vectorizer will skip stop words
vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text,
    max_features=2000,
)


# fit the vectorizer on the text
counts = vectorizer.fit_transform(training_data['review'])

# # get the vocabulary
vocab_dict = {k: v for k, v in vectorizer.vocabulary_.items()}
count_vocab_dict = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [count_vocab_dict[i] for i in range(len(count_vocab_dict))]


validation_data = pd.read_csv(
    'data/IMDB.csv', delimiter=',', nrows=10000, skiprows=30000 + 1)

test_data = pd.read_csv(
    'data/IMDB.csv', delimiter=',', nrows=20000, skiprows=40000 + 1)


def get_lang_vector_from_dataframe(frame):
    vectors = []
    for line in frame.iloc[:, 0]:
        line = clean_text(line).split()
        vec = [0 for _ in range(2000)]
        for word in line:
            try:
                vec[vocab_dict[word]] += 1
            except KeyError as e:
                pass

        vectors.append(vec)
    return vectors


training_vectors = get_lang_vector_from_dataframe(training_data)
validation_vectors = get_lang_vector_from_dataframe(validation_data)
test_vectors = get_lang_vector_from_dataframe(test_data)

# print(training_vectors)
# print(len(training_vectors))
# print(len(training_vectors[0]))
# print(len(vocabulary))

with open('data/IMDB_labels.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

data.pop(0)


def train_probability(is_positive=True, alpha=1):
    total_word_count = 0
    probs = [0] * len(vocabulary)  # [0, 0, 0, ...]
    for rev_idx, review in enumerate(training_vectors):  # Loop over the reviews
        if (data[rev_idx] == ['positive'] and not is_positive) or (data[rev_idx] == ['negative'] and is_positive):
            continue

        # review[i] == number of word off vocabulary[i] in the review
        # IE, if vocabulary[i] == "cheese" then review[i] is number of times "cheese" is in the review
        # Loop over the count of the i'th vocab word in said review
        for i, count_of_ith_vocab_word_in_review in enumerate(review):
            if count_of_ith_vocab_word_in_review > 0:
                probs[i] += count_of_ith_vocab_word_in_review
                total_word_count += count_of_ith_vocab_word_in_review

    for i in range(len(probs)):
        probs[i] += alpha
        probs[i] /= total_word_count + (len(vocabulary) * alpha)
        probs[i] = math.log(probs[i])

    return probs


alpha = 1
pos_probs = train_probability(alpha=alpha)
neg_probs = train_probability(is_positive=False, alpha=alpha)
total_pos_prob, total_neg_prob = learn_classes()

# print(pos_probs)
# print(neg_probs)
# print(total_pos_prob)  # p(y==1)
# print(total_neg_prob)  # p(y==0)

correct = 0
total = len(validation_vectors)
# For every review
for valid_idx, review in enumerate(validation_vectors):
    pos_prob = 1.0
    neg_prob = 1.0
    # For every vocab word in the review
    for i, count_of_ith_vocab_word_in_review in enumerate(review):
        if count_of_ith_vocab_word_in_review == 0:
            continue
        pos_prob *= pos_probs[i] * count_of_ith_vocab_word_in_review
        neg_prob *= neg_probs[i] * count_of_ith_vocab_word_in_review

    pos_prob /= total_pos_prob * 30000
    neg_prob /= total_neg_prob * 30000

    is_pos = pos_prob > neg_prob

    if is_pos and data[valid_idx + 30000][0] == 'positive':
        correct += 1
    elif not is_pos and data[valid_idx + 30000][0] == 'negative':
        correct += 1

print(correct / total)
