# 02-750: Automation of Biological Research, Spring 2019
# Homework 2 Question 2 - CAL Implementation

import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
from modAL.models import ActiveLearner


def GenerateSeparableData(num_samples):
	"""
	Generate 2D data for classification that are linearly separable using skicit learn make_blobs method
	Input:
	numsamples: the number of instances to generate
	Return:
	data: a 2D numpy array of size (numsamples, 2)
	trueLabels: a 1D numpy array of size (numsamples, ), the label is either 1 or 0
	"""

	data, true_labels = make_blobs(n_samples=num_samples, n_features=2, centers=2, cluster_std=0.7, random_state=0)
	# save the data as .npy file
	np.save('./data/data_cal.npy', data)
	np.save('./data/labels_cal.npy', true_labels)
	return data, true_labels


def PlotData(data, true_labels):
	"""
	Make scatter plot of the dataset, dots colored by class value
	"""
	colors = ['red' if l == 0 else 'blue' for l in true_labels]
	fig = plt.figure()
	plt.scatter(data[:, 0], data[:, 1], c=colors)
	plt.show()
	return


def CAL(clf, S, SLabels, U, ULabels):
	"""
	CAL algorithm as the query strategy for active learner
	Input:
	clf: classifier passed by ActiveLearner object.
	In your CAL implementation, you should not modify clf! Define new SVM estimators for fitting the data with label 1 and 0, respectively.
	S: the pool of labeled data
	SLabels: the label of the data in the pool of labeled data
	U: the pool of unlabeled data
	ULabels: the label of the data in the pool of unlabeled data
	Return:
	Should return the index of the query instance in the unlabeled pool, the label of the query instance (queried from oracle or inferred) as a numpy array, and a boolean flag indicating whether the instance is queried or inferred (True if queried from oracle, False if inferred).
	Note: At this step, please do not modify the labeled pool and unlabeled pool in this function. You will need to properly maintain the labeled pool and unlabeled pool of data in CALLearner() function where you will actually create the active learner and run the active learner.
	"""

	### TODO: Implement CAL algorithm
	S_copy_0, S_copy_1, SLabels_copy_0, SLabels_copy_1 = copy.deepcopy(S), copy.deepcopy(S), copy.deepcopy(SLabels), copy.deepcopy(SLabels)
	idx = np.random.randint(len(ULabels)) # we randomly pick one data point from the unlabeled pool to consider
	S_copy_0 = np.vstack((S_copy_0, U[idx].reshape(1, 2)))
	S_copy_1 = np.vstack((S_copy_1, U[idx].reshape(1, 2)))
	SLabels_0 = np.vstack((SLabels_copy_0, np.array([0])))
	SLabels_1 = np.vstack((SLabels_copy_1, np.array([1])))

	y_idx = ULabels[idx]
	is_queried = True

	clf_0 = SVC(kernel='linear', C=1000)
	clf_1 = SVC(kernel='linear', C=1000)
	clf_0.fit(S_copy_0, SLabels_0)
	learnable_0 = clf_0.score(S_copy_0, SLabels_0)
	clf_1.fit(S_copy_1, SLabels_1)
	learnable_1 = clf_1.score(S_copy_1, SLabels_1)

	if (learnable_0 == 1 and learnable_1 != 1):
		y_idx = np.array([0])
		is_queried = False

	if (learnable_1 == 1 and learnable_0 != 1):
		y_idx = np.array([1])
		is_queried = False

	if (learnable_1 != 1 and learnable_0 != 1):
		print("data is not separable!")
		is_queried = True

	if (learnable_0 == 1 and learnable_1 == 1):
		is_queried = True

	return idx, y_idx, is_queried


def CALLearner(X, y):
	"""
	Create an active learner with CAL query strategy and run the active learner on the given data set
	Input:
	The data set X and the corresponding labels
	Return:
	The accuracies evaluated on X, y using the fitted model with the labeled data so far whenever querying the true label of a data point from oracle as a one-demensional numpy array, the number of data points that are queried from oracle for the true label.
	"""

	# use SVM classifier with default parameters
	clf = SVC()
	# create an active learner with CAL as query strategy. The labeled pool of data is initially not empty, it contains two data points that belong to two classes.
	CAL_learner = ActiveLearner(estimator=clf, query_strategy=CAL, X_training=np.array([[0.5, 4.0], [2.0, 1.0]]), y_training=np.array([[0], [1]]))
	# In worst case, we would need to query all data points in the unlabeled pool.
	n_queries = len(y)

	# use variable i to keep track of the number of data points that are queried from oracle
	i = 0
	# store the accuracies evaluated on X, y whenever querying the true label of a data point from oracle
	accuracies = []

	### TODO: Write the main loop for running the CAL active learner, make sure you maintain the labeled pool and unlabeled pool properly, and calculate the accuracy of the estimater on all given data, i.e. X, y whenever you query a data point from the oracle for the true label.
	S, SLabels = np.array([[0.5, 4.0], [2.0, 1.0]]), np.array([[0], [1]])
	U, ULabels = copy.deepcopy(X), copy.deepcopy(y)

	while (i<n_queries and len(U) != 0):
		idx, y_idx, is_queried = CAL_learner.query(S, SLabels, U, ULabels)
		CAL_learner._add_training_data(U[idx].reshape(1, 2), y_idx.reshape(1,1))
		CAL_learner._fit_to_known()
		S, SLabels = np.vstack((S, U[idx].reshape(1, 2))),np.vstack((SLabels, y_idx.reshape(1,1)))
		U, ULabels = np.delete(U, idx, axis=0), np.delete(ULabels, idx)
		acc = CAL_learner.score(X, y)
		if (is_queried):
			i += 1
			accuracies.append(acc)

	return np.array(accuracies), i


def RandomQuery(clf, X_pool, n_instances=1):
	"""
	Query unlabeled data randomly
	Input:
	clf: classifier passed by ActiveLearner object
	X_pool: unlabed data instances to query from
	n_instances: number of queries to make
	Return:
	Should return the index of the query instance in the unlabeled pool, the data from X_pool at query_idx as a numpy array.
	"""
	### TODO: Implement random selection query strategy. This should be very simple, at most two lines of code.
	query_idx = np.random.randint(len(X_pool), size = n_instances)
	X_queried = X_pool[query_idx]

	return query_idx, X_pool[query_idx]


def RandomLearner(X, y):
	"""
	Create an active learner with random query strategy and run the active learner on the given data set. You should implement this also using modAL. Use SVM classifier with default parameter as the estimator.
	Input:
	The data set X and the corresponding labels
	Return:
	The accuracies evaluated on X, y whenever querying the true label of a data point from oracle as a one-demensional numpy array, the number of data points that are queried from oracle for the true label.
	"""

	random_learner = ActiveLearner(estimator=SVC(),
		query_strategy=RandomQuery,
		X_training=np.array([[0.5, 4.0], [2.0, 1.0]]),
		y_training=np.array([[0], [1]]))

	### TODO: Write the main loop for running the random active learner
	accuracies = []
	i = 0

	#S, SLabels = np.array([[0.5, 4.0], [2.0, 1.0]]), np.array([[0], [1]])
	U, ULabels = copy.deepcopy(X), copy.deepcopy(y)

	while (len(U) != 0):
		idx, instance = random_learner.query(U)
		i += 1
		random_learner._add_training_data(U[idx].reshape(1, 2), ULabels[idx].reshape(1, 1))
		random_learner._fit_to_known()
		U, ULabels = np.delete(U, idx, axis=0), np.delete(ULabels, idx, axis=0)
		acc = random_learner.score(X, y)
		accuracies.append(acc)

	return np.array(accuracies), i


def PlotAcc(acc_CAL, queries_CAL, acc_random, queries_random):
	"""
	Utility function for generating the accuracy plot for CAL and random active learning.
	Input: a one-dimensional numpy array of the calculated accuracies for CALLearner, the number of data points that are queried with CAL query strategy, a one-dimensional numpy array of the calculated accuracies for RandomLearner, the number of data points that are queried with random query strategy.
	"""

	### TODO: make the accuracy plot


	plt.plot(range(1,len(acc_CAL)+1), acc_CAL, label="CAL sampling")
	plt.plot(range(1,len(acc_random)+1), acc_random, label="Random Sampling")
	plt.legend()
	plt.xlabel("Query number")
	plt.ylabel("Test accuracy")
	plt.title("Test Accuracies in CAL & Random Sampling")
	plt.savefig("acc.png")
	plt.clf()

	names = ["CAL sampling", "Random Sampling"]
	values = [queries_CAL, queries_random]

	plt.scatter(names, values)
	plt.title("Number Queried")
	plt.savefig("queried.png")
	plt.clf()

	print(queries_CAL)
	return



if __name__ == '__main__':
	# The following line generates the dataset for this question. You do not need to run
	# data, labels = GenerateSeparableData(200)

	# load the generated synthetic data for classification that are linearly separable
	X = np.load('./data/data_cal.npy')
	y = np.load('./data/labels_cal.npy')
	# X, y are treated as the unlabeled pool of data

	# visualize the synthetic linearly separable 2D data
	# PlotData(X, y)

	# set random seed
	np.random.seed(66)

	PlotData(X, y)
	# run active learner with CAL query strategy
	acc_CAL, queries_CAL = CALLearner(X, y)

	# run random learner
	acc_random, queries_random = RandomLearner(X, y)

	# generate the plots of accuracies
	PlotAcc(acc_CAL, queries_CAL, acc_random, queries_random)
