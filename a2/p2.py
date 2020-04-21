'''
q2.py
Description:
    Train a multi-nomial Naive Bayes classifier with Laplace smooth with α = 1 on the training
    set. This involves learning P(y = 1), P(y = 0), P(wi|y = 1) for i = 1, ..., |V | and P(wi|y = 0) for
    i = 1, ..., |V | from the training data (the first 30k reviews and their associated labels).
'''

import sys
import numpy as np
import csv

'''
Learn p(y=1):
Given a set of N training reviews, MLE of the parameters are:
    P(y=1) = N1/N, where N1 is the number of positive reviews.
'''
def learn_classes():
    #learn P(y=1):
    with open('IMDB_labels.csv', newline='') as f:
        reader = csv.reader(f)
        data = list(reader)

    data.remove(data[0])
    print(data[0])
    count = 0
    for i in range(0, 30000):
        if data[i] == ['positive']:
            count += 1

    print(count)
    prob1 = count/30000
    print(prob1)

    #learn P(y=0):
    count2 = 30000 - count
    prob0 = count2/30000

    print(prob0)

#Function calls
learn_classes()
