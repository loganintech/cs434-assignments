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


# this vectorizer will skip stop words
vectorizer = CountVectorizer(
    stop_words="english",
    preprocessor=clean_text,
    max_features=2000,
)

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


# fit the vectorizer on the text
counts = vectorizer.fit(training_data)

# # get the vocabulary
vocab_dict = {k: v for k, v in vectorizer.vocabulary_.items()}
count_vocab_dict = {v: k for k, v in vectorizer.vocabulary_.items()}
vocabulary = [count_vocab_dict[i] for i in range(len(count_vocab_dict))]


def get_vectors_from_data(frame):
    vectors = []
    for line in frame:
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
    for rev_idx, review in enumerate(training_vectors):  # Loop over the reviews
        # If the review is positive, and we're not training positive, or if the review is negative, and we're not training negative, continue
        if training_labels[rev_idx] != is_positive:
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
        probs[i] = math.log(probs[i], 10)

    return probs


alpha = 1
pos_probs = train_model(is_positive=True,  alpha=alpha)
neg_probs = train_model(is_positive=False, alpha=alpha)
total_pos_prob, total_neg_prob, _, _ = learn_classes(training_labels)

# print(pos_probs)
# print(neg_probs)
# print(total_pos_prob)  # p(y==1)
# print(total_neg_prob)  # p(y==0)


def apply_model(x, is_positive=True):
    negative_term = 1
    positive_term = 1

    for i in range(len(vocabulary)):
        negative_term *= (neg_probs[i] ** x[i])
        positive_term *= (pos_probs[i] ** x[i])

    if is_positive:
        numerator = positive_term * total_pos_prob
    else:
        numerator = negative_term * total_neg_prob

    denominator = (negative_term * total_neg_prob) + \
        (positive_term * total_pos_prob)
    return numerator / denominator


def validate(vec, labels):
    _, _, pos, neg = learn_classes(labels)
    pos_corr = 0
    neg_corr = 0
    correct = 0
    for i, (x, posval) in enumerate(zip(vec, labels)):
        pos_prob = apply_model(x, is_positive=True)
        neg_prob = apply_model(x, is_positive=False)
        # print(pos_prob, neg_prob, pos_prob > neg_prob, posval)
        if pos_prob > neg_prob and posval:
            pos_corr += 1
            correct += 1
        elif pos_prob < neg_prob and not posval:
            neg_corr += 1
            correct += 1

    print("Total cor", correct / len(vec))
    print("Pos cor", pos_corr / pos)
    print("Neg cor", neg_corr / neg)


print("Validating training vectors")
validate(training_vectors, training_labels)
print("Validating validation vectors")
validate(validation_vectors, validation_labels)
