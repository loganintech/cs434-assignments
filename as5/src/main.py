import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
sns.set()

import argparse

from utils import load_data, f1, accuracy_score, load_dictionary, dictionary_info
from tree import DecisionTreeClassifier, RandomForestClassifier

def load_args():

	parser = argparse.ArgumentParser(description='arguments')
	parser.add_argument('--county_dict', default=1, type=int)
	parser.add_argument('--decision_tree', default=1, type=int)
	parser.add_argument('--random_forest', default=1, type=int)
	parser.add_argument('--ada_boost', default=1, type=int)
	parser.add_argument('--root_dir', default='../data/', type=str)
	args = parser.parse_args()

	return args


def county_info(args):
	county_dict = load_dictionary(args.root_dir)
	dictionary_info(county_dict)

def decision_tree_testing(x_train, y_train, x_test, y_test):
	print('Decision Tree\n\n')
	clf = DecisionTreeClassifier(max_depth=5)
	clf.fit(x_train, y_train)
	preds_train = clf.predict(x_train)
	preds_test = clf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = clf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))

def random_forest_testing(x_train, y_train, x_test, y_test):
	print('Random Forest\n\n')
	rclf = RandomForestClassifier(max_depth=5, max_features=10, n_trees=100)
	rclf.fit(x_train, y_train)
	preds_train = rclf.predict(x_train)
	preds_test = rclf.predict(x_test)
	train_accuracy = accuracy_score(preds_train, y_train)
	test_accuracy = accuracy_score(preds_test, y_test)
	print('Train {}'.format(train_accuracy))
	print('Test {}'.format(test_accuracy))
	preds = rclf.predict(x_test)
	print('F1 Test {}'.format(f1(y_test, preds)))
	
def create_trees(x_train, y_train, x_test, y_test):
	train = []
	test = []
	y = []
	for i in range(1, 26):
		y.append(i)
		print("LENGTH: " , i)
		h = DecisionTreeClassifier(max_depth=i)
		h.fit(x_train, y_train)
		preds_train = h.predict(x_train)
		preds_test = h.predict(x_test)
		train_accuracy = accuracy_score(preds_train, y_train)
		train.append(train_accuracy)
		test_accuracy = accuracy_score(preds_test, y_test)
		test.append(test_accuracy)
		print('Train {}'.format(train_accuracy))
		print('Test {}'.format(test_accuracy))
		preds = h.predict(x_test)
		print('F1 Test {}'.format(f1(y_test, preds)))
	return y, train, test

###################################################
# Modify for running your experiments accordingly #
###################################################
if __name__ == '__main__':
	args = load_args()
	x_train, y_train, x_test, y_test = load_data(args.root_dir)
	#if args.county_dict == 1:
		#county_info(args)
	#if args.decision_tree == 1:
		#decision_tree_testing(x_train, y_train, x_test, y_test)
	if args.random_forest == 1:
		random_forest_testing(x_train, y_train, x_test, y_test)


	#####ADDED VARIABLES####
	plotValues = False	# set to true to generate images
	setting = 0 		# 0 = depth testing, 1 = tree testing, 2 = feature testing
	########################

	if plotValues == True:

		a_data_training = []
		a_data_test = []
		a_data_f1 = []
		if setting == 0:
			m_depth = [2, 5, 10, 15, 20, 25, 30, 35, 40]
			for i in m_depth:
				clf = DecisionTreeClassifier(max_depth=i)
				clf.fit(x_train, y_train)
				preds_train = clf.predict(x_train)
				preds_test = clf.predict(x_test)
				a_data_training.append(accuracy_score(preds_train, y_train))
				a_data_test.append(accuracy_score(preds_test, y_test))
				preds = clf.predict(x_test)
				a_data_f1.append(f1(y_test, preds))

			plt.plot(m_depth, a_data_training, label='training')
			plt.plot(m_depth, a_data_test, label='test')
			plt.plot(m_depth, a_data_f1, label='F1')
			plt.xlabel('number of trees')
			plt.ylabel('accuracy')
			plt.title('accuracy by number of depths')
			plt.legend()
			plt.savefig("Q1_E.png")

		if setting == 1:
			n_tree = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
			for i in n_tree:
				rclf = RandomForestClassifier(max_depth=7, max_features=11, n_trees=i)
				rclf.fit(x_train, y_train)
				preds_train = rclf.predict(x_train)
				preds_test = rclf.predict(x_test)
				a_data_training.append(accuracy_score(preds_train, y_train))
				a_data_test.append(accuracy_score(preds_test, y_test))
				preds = rclf.predict(x_test)
				a_data_f1.append(f1(y_test, preds))

			plt.plot(n_tree, a_data_training, label='training')
			plt.plot(n_tree, a_data_test, label='test')
			plt.plot(n_tree, a_data_f1, label='F1')
			plt.xlabel('number of trees')
			plt.ylabel('accuracy')
			plt.title('accuracy by number of trees')
			plt.legend()
			plt.savefig("Q2_A.png")
		if setting == 2:
			n_features = [1, 2, 5, 8, 10, 20, 25, 35, 50]
			for i in n_features:
				rclf = RandomForestClassifier(max_depth=7, max_features=i, n_trees=50)
				rclf.fit(x_train, y_train)
				preds_train = rclf.predict(x_train)
				preds_test = rclf.predict(x_test)
				a_data_training.append(accuracy_score(preds_train, y_train))
				a_data_test.append(accuracy_score(preds_test, y_test))
				preds = rclf.predict(x_test)
				a_data_f1.append(f1(y_test, preds))

			plt.xlabel('number of features')
			plt.ylabel('accuracy')
			plt.title('accuracy by number of features')
			plt.plot(n_features, a_data_training, label='training')
			plt.plot(n_features, a_data_test, label='test')
			plt.plot(n_features, a_data_f1, label='F1')
			plt.legend()
			plt.savefig("Q2_C.png")
			plt.show()


	print('Done')
	
	





