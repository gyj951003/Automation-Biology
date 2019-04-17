from modAL.models import ActiveLearner, Committee
from modAL.uncertainty import uncertainty_sampling, margin_sampling, entropy_sampling
from modAL.disagreement import vote_entropy_sampling
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy

def load_setA(fname):
	'''Loads and properly formats data from Image data (set A)
	:param fname: Location and name of data file
	:return X: array (500x26)
	:return y: array (500,)'''

	data = pd.read_csv(fname)
	X = data.iloc[:,:26].values
	y = LabelEncoder().fit_transform(data.iloc[:,-1])
	return X, y

def split_data(X,y,random_state=0):
	'''Splits data into training, "unlabeled" pools, and test sets
	Training data consists of single instance for each label
	:param X: data matrix
	:param y: labels
	:return X_training: subset of X for training
	:return y_training: subset of y for training
	:return X_pool: subset of X used as unlabeled pool
	:return y_pool: real labels for X_pool
	:return X_test: subset of X for testing
	:return y_test: subset of y for testing'''

	X_training, y_training = [], []
	for i in range(10):
		unique_label_idx = list(y).index(i)
		xx = X[unique_label_idx]
		X_training.append(xx)
		y_training.append(i)
		X, y = np.delete(X, unique_label_idx, axis=0), np.delete(y, unique_label_idx, axis=0)

	X_pool, X_test, y_pool, y_test = train_test_split(X,y,test_size=0.5,random_state=random_state)

	return np.asarray(X_training), X_pool, np.asarray(y_training), y_pool, X_test, y_test

def random_query(clf,X_pool,n_instances=1):
	'''Query unlabeled data randomly.
	:param clf: Classifier passed by ActiveLearner object
	:param X_pool: Unlabeled data instances to query from
	:param n_instance: Number of queries to make
	:returns query_indx: numpy array of selected query indices
	:returns X_queried: numpy array of data from X_pool at indices query_indx'''

	###TODO: implement random query
	query_indx = np.random.randint(len(X_pool))
	X_queried = X_pool[query_indx]

	return query_indx, X_queried

def active_learner(estimator,X_training,y_training,query_strategy=uncertainty_sampling,random_state=0):
	'''Instantiate an ActiveLearner object within modAL
	:param estimator: the underlying classifier used for learning
	:param query_strategy: a function that define how to make queries to the oracle
	:param X_training: initial training data
	:param y_training: labels
	:return ActiveLearner(): Active learner object'''

	###TODO: Correctly call ActiveLearner using modAL
	learner = ActiveLearner (
		estimator=estimator,
    	query_strategy=query_strategy,
    	X_training=X_training, y_training=y_training
	)

	return learner #change this to return active learner

def pool_based(X,y,n_queries=50,random_state=0):
	'''Pool based active learning compairing uncertainty sampling, entropy sampling,
	   margin sampling, and random sampling query strategies.
	:param X: Data matrix
	:param y: labels
	:param n_queries: number of calls to the oracle
	:param random_state: sets random seed
	:returns uncertainty_acc: accuracy after final query using uncertainty sampling
	:returns entropy_acc: accuracy after final query using entropy sampling
	:returns margin_acc: accuracy after final query using margin sampling
	:returns random_acc: accuracy after final query using random sampling  '''


	X_training, X_pool_uncertainty, y_training, y_pool_uncertainty, X_test, y_test = split_data(X,y)

	#suggested way to make copies of unlabeled pools for each sampling type
	X_pool_margin, y_pool_margin = copy.deepcopy(X_pool_uncertainty), copy.deepcopy(y_pool_uncertainty)
	X_pool_entropy, y_pool_entropy = copy.deepcopy(X_pool_uncertainty), copy.deepcopy(y_pool_uncertainty)
	X_pool_random, y_pool_random = copy.deepcopy(X_pool_uncertainty), copy.deepcopy(y_pool_uncertainty)

	###TODO: Pool based sampling experiments using uncertainty sampling, entropy sampling, margin sampling, and random sampling
	learner_uncertainty = active_learner(RandomForestClassifier(random_state = random_state),X_training,y_training, uncertainty_sampling, random_state)
	uncertainty_acc, _ = learn(learner_uncertainty, n_queries, X_pool_uncertainty, y_pool_uncertainty, X_test, y_test)

	learner_entropy = active_learner(RandomForestClassifier(random_state = random_state),X_training,y_training, entropy_sampling, random_state)
	entropy_acc, _ = learn(learner_entropy, n_queries, X_pool_margin, y_pool_margin, X_test, y_test)

	learner_margin = active_learner(RandomForestClassifier(random_state = random_state),X_training,y_training, margin_sampling, random_state)
	margin_acc, _ = learn(learner_margin, n_queries, X_pool_entropy, y_pool_entropy, X_test, y_test)

	learner_random = active_learner(RandomForestClassifier(random_state = random_state),X_training,y_training, random_query, random_state)
	random_acc, _ = learn(learner_random, n_queries, X_pool_random, y_pool_random, X_test, y_test)

	print(uncertainty_acc)
	print(entropy_acc)
	print(margin_acc)
	print(random_acc)
	return uncertainty_acc, entropy_acc, margin_acc, random_acc

def learn(learner, n_queries, X, y, X_test, y_test, cfs_mtx=False):
	if (cfs_mtx):
		cfs_matrices = list()

	acc_list = list()

	for idx in range(n_queries):
		query_idx, query_instance = learner.query(X)
		learner.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, ))
		X, y = np.delete(X, query_idx, axis=0), np.delete(y, query_idx)

		if (cfs_mtx and idx == 0):
			for sublearner in learner:
				y_pred = sublearner.predict(X_test)
				cfs_matrices.append(confusion_matrix(y_test, y_pred))

		acc = learner.score(X_test, y_test)
		acc_list.append(acc)

	acc = learner.score(X_test, y_test)

	if (cfs_mtx):
		return acc, acc_list, cfs_matrices

	return acc, acc_list


def committee(X,y,n_learners=3,n_queries=50):
	'''Query by committee
	:param X: Data matrix
	:param y: labels
	:param n_learners: number of committee members
	:param n_queries: number of calls to the oracle
	:returns max_cost_idx: which query yield largest difference in test accuracy betweed the two committees
	:returns entropy_acc: Accuracy of entropy sampling committee at query max_cost_idx
	:returns random_acc: Accuracy of random sampling committee at query max_cost_idx'''

	X_training, X_pool_entropy, y_training, y_pool_entropy, X_test, y_test = split_data(X,y)

	X_pool_random, y_pool_random = copy.deepcopy(X_pool_entropy), copy.deepcopy(y_pool_entropy)

	###TODO: Query by committee. Compare vote entropy sampling with random sampling
	learner_entropy_list = list()
	learner_random_list = list()
	for seed in range(n_learners):
		learner_entropy = active_learner(RandomForestClassifier(random_state = seed), X_training, y_training, vote_entropy_sampling, seed)
		learner_entropy_list.append(learner_entropy)
		learner_random = active_learner(RandomForestClassifier(random_state = seed), X_training, y_training, random_query, seed)
		learner_random_list.append(learner_random)

	committee_entropy = Committee(learner_list = learner_entropy_list, query_strategy = vote_entropy_sampling)
	_, acc_list_entropy, confusion_matrices = learn(committee_entropy, n_queries, X_pool_entropy, y_pool_entropy, X_test, y_test, True)

	committee_random = Committee(learner_list = learner_random_list, query_strategy = random_query)
	_, acc_list_random = learn(committee_random, n_queries, X_pool_random, y_pool_random, X_test, y_test)

	for m in confusion_matrices:
		print(m)

	acc_list = [acc_list_entropy, acc_list_random]

	plots(acc_list, ["Uncertainty sampling", "Random Sampling"], "Committee Test Accuracy Over Queries", "committee_accuracies.png")

	diff = np.asarray(acc_list_entropy) - np.asarray(acc_list_random)
	diff = np.absolute(diff)
	max_cost_idx = np.argmax(diff)
	entropy_acc = acc_list_entropy[max_cost_idx]
	random_acc = acc_list_random[max_cost_idx]
	return max_cost_idx, entropy_acc, random_acc


def plots(accuracies,query_labels,title,outfile):
	'''
	Create plots.  This code serves as an example on how to use matplotlib for those new to python.
	Feel free to use this in creating your plots, you may also write your own code if you are more
	familiar with plotting using python and have some method that is more natural or suitable to your
	implementations above.  Its designed to be called individually for each plot you want to make

	:param accuracies: List of lists (or num_learners x num_queries dimension numpy array) test accuracies after each call to the oracle for each query method
	:param query_labels: List of str that states which query style is being plotted. query_labels[i] should correspond to accuracies[i]
	eg. ["Uncertainty sampling", "Random Sampling"]
	:param title: str A name for your plot!
	:param outfile: str name of output file to save to
	'''

	x_axis = range(1,len(accuracies[0])+1)

	for acc, q_label in zip(accuracies, query_labels):
		plt.plot(x_axis, acc, label=q_label)
		plt.legend()
	plt.xlabel("Query number")
	plt.ylabel("Test accuracy")
	plt.title(title)
	plt.savefig(outfile)
	plt.clf()


if __name__ == '__main__':

	X,y = load_setA("Data.csv")

	uncertainty_acc, entropy_acc, margin_acc, random_acc = pool_based(X,y)

	cost, e, r = committee(X,y)
