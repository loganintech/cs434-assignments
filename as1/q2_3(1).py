import sys
import numpy as np
import math as mth
import csv
import matplotlib.pyplot as plt
from numpy import linalg as LA


def loadCSV(fileName,nFeat):
	xload = []
	yload = []
	temp = []

	with open(fileName) as f:
		string = f.read().replace('\n',',')
		lines = string.split(',')									#lines = list of every data point

	lines = lines[:-1]

	for count in range (0,len(lines)):
		if (count+1) % (nFeat+1) == 0:							#if position is 14(y value) append to y
			yload.append([float(lines[count])])
		else:														#else (if first position append a dummy '1') append to temp
			temp.append(float(lines[count]))
    
	xload = [temp[i:i+nFeat] for i in range(0,len(temp),nFeat)]		#splitting x into a list of lists separated by each instance
	x1 = np.matrix(xload)											#converting x from list of lists to matrices
	y1 = np.matrix(yload)											#converting y from list of lists to matrices
	x1 = x1 * (1.0/255.0)       									#converts x values into percentages

	return x1,y1


def calculate_y_hat(w, x):
    exponent = (-1*np.dot(w, x.T))
    new_exponent = float(exponent.item(0))
    y_hat = 1./(1. + mth.exp(new_exponent))
    return y_hat


def check_accuracy(w, x, y):
    correct = 0
    for i in range(0, x.shape[0]):
        if y[i] == 1:
            sigmoid = calculate_y_hat(w, x[i])
            if sigmoid >= 0.5:
                correct += 1
        elif y[i] == 0:
            sigmoid_complement = 1 - calculate_y_hat(w, x[i])
            if sigmoid_complement >= 0.5:
                correct += 1

    percentage_correct = correct/x.shape[0]
    return percentage_correct

#********************************************************************
#NEED TO FIX
#********************************************************************
def calculate_l(w, x, y):
    #Calculate the loss for the ith example using for loop from range (0, x.shape[0])
    #Then calculate the total and return that
    l = 0
    term1 = 0
    term2 = 0

    term1 = calculate_y_hat(w, x)
    term1 = np.log10(term1)
    term1 = y * term1
    term1 = (-1) * term1

    term2 = 1 - (calculate_y_hat(w, x))
    term2 = np.log10(term2)
    term2 = (1 - y) * term2

    l = term1 - term2

    return l



#**************************************************************************
    #LOGISTIC REGRESSION FUNCTION
#**************************************************************************
def gradient(reg_strength):
    training_accuracies = []
    testing_accuracies = []

    x,y = loadCSV("usps_train.csv", 256)
    x1, y1 = loadCSV("usps_test.csv", 256)
    w = np.matrix([0] * 256)

    learning_rate = 0.001 #let the learning rate be 0.001
    iterations = 0 #number of times it has iterated through the while loop

    reg_term = 0

    while True:
        gradient = np.matrix([0] * 256)
        for i in range(1,x.shape[0]):
            y_hat = calculate_y_hat(w, x[i])
            a = y_hat - y[i].item(0)
            b = a*x[i]
            gradient = np.add(gradient, b)
            # CALCULATE NORM(w)^2 * lambda * 1/2
            reg_term = LA.norm(w)
            reg_term = reg_term ** 2
            reg_term = reg_strength * reg_term
            reg_term = reg_term / 2
            gradient = np.add(gradient, reg_term)

        result = learning_rate * gradient
        w = np.subtract(w, result)

        #training_accuracies.append(check_accuracy(w, x, y))
        #testing_accuracies.append(check_accuracy(w, x1, y1))

        iterations += 1
        if iterations == 100:
            break

    return x, y, w
    #return training_accuracies, testing_accuracies



#*******************************************************#
#*******************************************************#
#*******************************************************#

#Array of reg_strengths: ranges from 10^-3 to 10^3
reg_strength = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
l = []


for i in range(0, 6):
    x, y, w = gradient(reg_strength[i])
    l.append(calculate_l(w, x, y)

print(l)

#Number of iterations
#iterations = []

#Filling in number of iterations
#for i in range(0,100):
#    iterations.append(i+1)

#Graphing:
#gradient(0)
#plt.ylabel("Accuracy")
#plt.xlabel("Iteration")
#plt.title("Accuracy as  Function of Iteration")
#plt.plot(iterations, training_accuracies, 'b', label='training')
#plt.plot(iterations, testing_accuracies, 'r', label='testing')
#plt.legend()
#plt.show()
#plt.savefig(f"graph_results.png")
