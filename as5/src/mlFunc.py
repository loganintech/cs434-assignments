from sklearn.feature_extraction.text import CountVectorizer
import re
import csv
import math
import matplotlib.pyplot as plt
import numpy as np


with open('./data/train.csv', "r", encoding="utf-8") as f:
    line = csv.reader(f)
    textID = []
    text = []
    selText = []
    mood = []
    for row in line:
        textID.append(row[0])
        text.append(row[1])
        selText.append(row[2])
        mood.append(row[3])
        print(row[0])


    #without_header = lines
    #training_data = without_header[:30000]
    #validation_data = without_header[30000:40000]
    #test_data = without_header[40000:]
    #assert len(training_data) == 30000
    #assert len(validation_data) == 10000
    #assert len(test_data) == 10000
'''
with open('data/IMDB_labels.csv', "r", encoding='utf-8') as f:
    lines = f.readlines()
    lines = list(map(lambda line: line.split(",")[0], lines[1:]))
    training_labels = lines[:30000]
    validation_labels = lines[30000:]
    assert len(training_labels) == 30000
    assert len(validation_labels) == 10000
    assert len(lines[40000:]) == 0
'''



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
    # Note, left out . to allow phrases
    filters = '!"\'#$%&()*+,-/:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

def learn_classes(labels):

    # Track our positive count
    positive_count = 0
    negative_count = 0
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

if __name__ == "__main__":
    print("done")
    print(len(text))
    print(len(textID))
    print(len(selText))
    print(len(mood))
