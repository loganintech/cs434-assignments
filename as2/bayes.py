import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re


def clean_text(text):

    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)
    text = re.sub(r"\'", "", text)
    text = re.sub(r"\"", "", text)
    #pattern = r'[^a-zA-z0-9\s]'
    #text = re.sub(pattern, '', text)

    # convert text to lowercase
    text = text.strip().lower()

    # replace punctuation characters with spaces
    filters = '!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text


# Importing the dataset
training_data = pd.read_csv('data/IMDB.csv', delimiter=',', nrows=30001)
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


validation_vectors = get_lang_vector_from_dataframe(validation_data)
test_vectors = get_lang_vector_from_dataframe(test_data)
