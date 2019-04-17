# 02-750: Automation of Biological Research, Spring 2019
# Homework 3 Question 2 - DHM Implementation

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
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

    data, true_labels = make_blobs(n_samples=num_samples, n_features=2, centers=2, cluster_std=1.2, random_state=0)
    # save the data as .npy file
    np.save('./data/data_DHM.npy', data)
    np.save('./data/labels_DHM.npy', true_labels)
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


def DHM(clf, S, SLabels, T, TLabels, U, ULabels):
    """
    DHM algorithm as the query strategy for active learner
    Input:
    clf: classifier passed by ActiveLearner object.
    In your DHM implementation, you should not modify clf! Define new SVM estimators for fitting the data with label 1 and 0, respectively.
    S: the pool of labeled data with inferred label
    SLabels: the label of the data in the pool of labeled data with inferred label
    T: the pool of labeled data with queried label
    TLabels: the label of the data in the pool of labeled data with queried label
    U: the pool of unlabeled data
    ULabels: the label of the data in the pool of unlabeled data
    Return:
    Should return the index of the query instance in the unlabeled pool, the label of the query instance (queried from oracle or inferred) as a numpy array, and a boolean flag indicating whether the instance is queried or inferred (True if queried from oracle, False if inferred).
    Note: At this step, please do not modify the labeled pool and unlabeled pool in this function. You will need to properly maintain the labeled pool and unlabeled pool of data in DHMLearner() function where you will actually create the active learner and run the active learner. To learn a classifier consistent with S and minimizes error on T, when fit the SVM estimator, set different sample weights for data in different pools. For this assignment, set the sample weights of the data in S as 1e9, set the sample weights of the data in T as 1.
    """

    idx = np.random.randint(len(ULabels))
    ST = copy.deepcopy(S)
    STLabel = copy.deepcopy(SLabels)

    S_copyT = np.vstack((copy.deepcopy(S), U[idx, :].reshape(-1, 2)))
    SLabelsPlusT = np.vstack((copy.deepcopy(SLabels), np.array(1)))
    SLabelsMinusT = np.vstack((copy.deepcopy(SLabels), np.array(0)))

    weights = np.ones(len(SLabelsPlusT))
    weights = weights * 1e9

    if TLabels is not None:
        ST = np.vstack((ST, T))
        STLabel = np.vstack((STLabel, copy.deepcopy(TLabels)))

        S_copyT = np.vstack((S_copyT, T))
        SLabelsPlusT = np.vstack((SLabelsPlusT, copy.deepcopy(TLabels)))
        SLabelsMinusT = np.vstack((SLabelsMinusT, copy.deepcopy(TLabels)))

        weights = np.append(weights, np.ones(len(TLabels)))

    # TODO: Implement DHM algorithm
    classifierPlus = SVC(gamma='scale')
    classifierMinus = SVC(gamma='scale')

    # learn with S_copyT|SLabelsPlusT, S_copyT\SLabelsMinusT
    classifierPlus = classifierPlus.fit(S_copyT, SLabelsPlusT, sample_weight=weights)
    classifierMinus = classifierMinus.fit(S_copyT, SLabelsMinusT, sample_weight=weights)

    # error calculted on ST|STLabel
    errorPlus = 1.0 - classifierPlus.score(ST, STLabel)
    errorMinus = 1.0 - classifierMinus.score(ST, STLabel)

    # Calculate error bound delta
    t = len(S_copyT)
    beta = 0.11 * np.sqrt(2 * np.log(t)/t)
    delta = beta**2 + beta * (np.sqrt(errorPlus) + np.sqrt(errorMinus))

    is_queried = False

    if (errorPlus - errorMinus > delta):
        label = np.array(0)
    elif (errorMinus - errorPlus > delta):
        label = np.array(1)
    else:
        label = ULabels[idx]
        is_queried = True

    return idx, label, is_queried


def DHMLearner(X, y):
    """
    Create an active learner with DHM query strategy and run the active learner on the given data set
    Input:
    The data set X and the corresponding labels
    Return:
    The accuracies evaluated on X, y using the fitted model with the labeled data so far whenever querying the true label of a data point from oracle as a one-demensional numpy array, the number of data points that are queried from oracle for the true label.
    """

    # use SVM classifier with default parameters
    clf = SVC(gamma='scale')
    # create an active learner with DHM as query strategy. The S and SLabels pool are initially not empty, it contains two data points that belong to two classes.
    DHM_learner = ActiveLearner(estimator=clf, query_strategy=DHM, X_training=np.array([[0.5, 4.0], [2.0, 1.0]]), y_training=np.array([[0], [1]]))
    # In worst case, we would need to query all data points in the unlabeled pool.
    n_queries = len(y)
    # use variable i to keep track of the number of data points that are queried from oracle
    i = 0
    # store the accuracies evaluated on X, y whenever querying the true label of a data point from oracle
    accuracies = []
    # we create another active learner, and simply use it to maintain the T and TLabels pool. Do not use this active learner to fit.
    DHM_queried = ActiveLearner(estimator=clf)

    # TODO: Write the main loop for running the DHM active learner, make sure you maintain the labeled pool, both S/SLabels and T/TLabels, and unlabeled pool properly, and calculate the accuracy of the estimater on all given data, i.e. X, y whenever you query a data point from the oracle for the true label.
    while i < n_queries:
        if len(DHM_learner.y_training) == 2:
            U = X
            ULabels = y
            S = np.array([[0.5, 4.0], [2.0, 1.0]])
            SLabels = np.array([[0], [1]])
            T = None
            TLabels = None
        else:
            if is_queried:
                if TLabels is None:
                    T = U[query_idx:query_idx+1]
                    TLabels = ULabels[query_idx:query_idx+1]
                else:
                    T = np.vstack((T, U[query_idx:query_idx+1]))
                    TLabels = np.vstack((TLabels, ULabels[query_idx:query_idx+1]))
            else: # not queried but inferred
                S = np.vstack((S, U[query_idx:query_idx+1]))
                SLabels = np.vstack((SLabels, ULabels[query_idx:query_idx+1]))

            U = np.delete(U, query_idx, axis=0)
            ULabels = np.delete(ULabels, query_idx)
            if not len(U):
                break


        query_idx, query_label, is_queried = DHM_learner.query(S, SLabels, T, TLabels, U, ULabels)
        # add to training data
        DHM_learner._add_training_data(U[query_idx, :].reshape(-1, 2), query_label.reshape(-1, 1))
        # fit on training data
        DHM_learner._fit_to_known()
        # calculate the accuracy of the learned estimator on the entire dataset
        acc = DHM_learner.score(X, y)

        if is_queried:
            accuracies.append(acc)
            i += 1

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
    n_samples = len(X_pool)
    query_idx = np.random.choice(range(n_samples))
    return query_idx, X_pool[query_idx]


def RandomLearner(X, y):
    """
    Create an active learner with random query strategy and run the active learner on the given data set. You should implement this also using modAL. Use SVM classifier with default parameter as the estimator.
    Input:
    The data set X and the corresponding labels
    Return:
    The accuracies evaluated on X, y whenever querying the true label of a data point from oracle as a one-demensional numpy array, the number of data points that are queried from oracle for the true label.
    """
    random_learner = ActiveLearner(estimator=SVC(gamma='scale'),
        query_strategy=RandomQuery,
        X_training=np.array([[0.5, 4.0], [2.0, 1.0]]),
        y_training=np.array([[0], [1]]))

    accuracies = []
    n_queries = len(y)
    i = 0
    while i < n_queries:
        if len(random_learner.y_training) == 2:
            U = X
            ULabels = y
        else:
            U = np.delete(U, query_idx, axis=0)
            ULabels = np.delete(ULabels, query_idx)
            if not len(U):
                break

        query_idx, query_instance = random_learner.query(U)

        # add to training data
        random_learner._add_training_data(U[query_idx, :].reshape(-1, 2), ULabels[query_idx].reshape(-1, 1))
        # fit on training data
        random_learner._fit_to_known()
        # calculate the accuracy of the learned estimator on the entire dataset
        accuracies.append(random_learner.score(X, y))
        i += 1

    return np.array(accuracies), i


def PlotAcc(acc_DHM, queries_DHM, acc_random, queries_random):
    """
    Utility function for generating the accuracy plot for DHM and random active learning.
    Input: a one-dimensional numpy array of the calculated accuracies for DHMLearner, the number of data points that are queried with DHM query strategy, a one-dimensional numpy array of the calculated accuracies for RandomLearner, the number of data points that are queried with random query strategy.
    """

    # TODO: make the accuracy plot
    plt.plot(range(1, queries_DHM+1), acc_DHM, label="DHM sampling")
    plt.plot(range(1, queries_random+1), acc_random, label="Random Sampling")
    plt.legend()
    plt.xlabel("Query number")
    plt.ylabel("Test accuracy")
    plt.title("Test Accuracies in DHM & Random Sampling")
    plt.savefig("acc.png")
    plt.clf()

    print(queries_DHM, queries_random)
    return



if __name__ == '__main__':
    # The following line generates the dataset for this question. You do not need to run
    # data, labels = GenerateSeparableData(200)

    # load the generated synthetic data for classification that are linearly separable
    X = np.load('./data/data_DHM.npy')
    y = np.load('./data/labels_DHM.npy')
    # X, y are treated as the unlabeled pool of data

    # visualize the synthetic linearly separable 2D data
    # PlotData(X, y)

    # set random seed
    np.random.seed(666)

    # run active learner with DHM query strategy
    acc_DHM, queries_DHM = DHMLearner(X, y)

    # run random learner
    acc_random, queries_random = RandomLearner(X, y)

    # generate the plots of accuracies
    PlotAcc(acc_DHM, queries_DHM, acc_random, queries_random)
