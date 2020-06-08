from sklearn.feature_extraction.text import CountVectorizer
import re
import csv
import math
import matplotlib.pyplot as plt
import numpy as np

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
    filters = '!"\'#$%&()+,-/:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text

with open('./data/train.csv', "r", encoding="utf-8") as f:
    line = csv.reader(f)
    textID = []
    text = []
    selText = []
    mood = []
    pos = 0
    neg = 0
    neu = 0
    for row in line:
        textID.append(row[0])
        text.append(row[1])
        selText.append(clean_text(row[2]))
        mood.append(row[3])
        if row[3] == "positive":
            pos += 1
        elif row[3] == "neutral":
            neu += 1
        elif row[3] == "negative":
            neg += 1

        print(clean_text(row[2]) + " , " + row[3])




if __name__ == "__main__":
    print("done")
    print(len(text))
    print(len(textID))
    print(len(selText))
    print(len(mood))
    print(pos)
    print(neu)
    print(neg)
    print(pos/(pos+neu+neg))
    print(neu/(pos+neu+neg))
    print(neg/(pos+neu+neg))