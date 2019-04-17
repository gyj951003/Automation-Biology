import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from modAL.models import ActiveLearner
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import scipy.stats as stats
from itertools import groupby


def VariableCostOracle(data):
    '''
    Pre-compute the "price list" of the variable cost oracle
    '''
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', probability=True)
    X = data.iloc[:, data.columns != 'Label']
    Y = data['Label']
    clf.fit(X, Y)
    y_prob = clf.predict_proba(X)
    class_num = np.unique(Y).shape[0]
    cost = 1 - (np.max(y_prob, axis=1) - 1.0 / class_num) / (1 - 1.0 / class_num)
    # save cost of variable cost oracle to .npy file
    np.save('./cost.npy', cost)
    return cost

def ProactiveQuery(clf, L_X, L_y, UL_X, UL_y, cost_non_unif, cost_unif, mode):
    '''
    Proactive learning strategy
    clf: classifier passed by ActiveLearner object.
    L_X: the pool of labeled data
    L_y: the labels of the data in the labeled data pool
    UL_X: the pool of unlabeled data
    UL_y: the labels of the data in the unlabeled data pool
    cost_non_unif: the "price list" of the variable cost oracle
    cost_unif: the cost of the uniform cost oracle as a scalar
    mode: 'proactive' for proactive learning query strategy,
        'uniform' for only querying from the uniform cost oracle
        'random' for randomly querying an instance from a randomly selected oracle
    Return: the index of the queried instance and the corresponding oracle
    '''

    # TODO:
    # train SVM on labeled data
    classifier = svm.SVC(gamma='scale', decision_function_shape='ovo',
                         probability=True)
    # compute the utilities for each unlabeled instance for both variable cost oracle and uniform cost oracle
    classifier.fit(L_X, L_y.reshape(-1, 1))
    prob = classifier.predict_proba(UL_X)
    # compute V as uncertainty query selection criterion using entropy

    V = np.zeros(UL_X.shape[0])

    for i in range(UL_X.shape[0]):
        V[i] = stats.entropy(prob[i])
    # normalize V so that it is in the same range as cost
    V = V / np.max(V)
    # Calculate the utilize function
    U1 = V - cost_unif
    U2 = V - cost_non_unif

    x_star, k_star = None, None
    # implement the three modes, and return x_star, k_star

    if (mode == 'uniform'):
        k_star = 1
        x_star = np.argmax(U1)

    if (mode == 'random'):
        k_star = np.random.randint(1, 3)
        x_star = np.random.randint(0, UL_X.shape[0])
        #if (k_star == 1):
        #    x_star = np.argmax(U1)
        #else:
        #    x_star = np.argmax(U2)

    if (mode == 'proactive'):
        max_1 = np.argmax(U1)
        max_2 = np.argmax(U2)
        if (U1[max_1] > U2[max_2]):
            k_star = 1
            x_star = max_1
        else:
            k_star = 2
            x_star = max_2

    return x_star, k_star


def ProactiveLearning(train_data, test_data, budget, cost_non_unif, cost_ratio, class_num, mode):
    '''
    Create an active learner with proactive query strategy and run the active learner on the given data set
    train_data: the train data, which is used to do proactive learning
    test_data: the held-out test data, which is used to compute classification accuracy
    budget: total amount of prices that is allowed to pay
    cost_non_unif: the "price list" of the variable cost oracle
    cost_ratio: defined as mean(cost_non_unif) / cost_unif
    class_num: the number of classes of the data set, for this data, it is 10
    mode: 'proactive' for proactive learning query strategy,
        'uniform' for only querying from the uniform cost oracle
        'random' for randomly querying an instance from a randomly selected oracle
    Return: a list of accuracies, a list of cumulative costs, a list of queried oracles,
        each element corresponds to each iteration.
    '''

    # cost of uniform cost oracle
    cost_unif = np.mean(cost_non_unif) / cost_ratio

    # use SVM classifier
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', probability=True)
    # load the initial, free labeled data
    initial_labeled_data = np.load('./initial_labeled_sample.npy')
    L_X = initial_labeled_data[:, :-1]
    L_y = initial_labeled_data[:, -1].reshape(-1, 1)
    # create an active learner with proactive learning strategy
    learner = ActiveLearner(estimator=clf,
        query_strategy=ProactiveQuery,
        X_training=initial_labeled_data[:, :-1],
        y_training=initial_labeled_data[:, -1].reshape(-1, 1))

    # Initially, the unlabeled pool of data is the entire train data
    UL_X, UL_y = train_data.iloc[:, data.columns != 'Label'].values, train_data['Label'].values

    test_X, test_y = test_data.iloc[:, data.columns != 'Label'].values, test_data['Label'].values

    accuracy = []
    total_cost = [0] # here a dummy cost of 0 is added for convenience
    oracle = []

    while total_cost[-1] < budget and UL_X.shape[0] != 0:

        # TODO: implement the active learning loop with proactive learning query strategy
        x_star, k_star = learner.query(L_X, L_y, UL_X, UL_y, cost_non_unif, cost_unif, mode)
        learner.teach(UL_X[x_star:x_star+1], UL_y[x_star:x_star+1].reshape(-1, 1))
        score = learner.score(test_X, test_y)
        accuracy.append(score)
        oracle.append(k_star)

        if (k_star == 1):
            total_cost.append(total_cost[-1] + cost_unif)
        else: # query from non_uninf oracle
            total_cost.append(total_cost[-1] + cost_non_unif[x_star])

        #print(UL_X.shape[0], x_star, total_cost[-1])

        # Add x_star to L_X, L_Y
        L_X = np.append(L_X, UL_X[x_star:x_star+1], axis = 0)
        L_y = np.append(L_y, UL_y[x_star].reshape(-1, 1), axis = 0)

        # Delete x_star from UL_X, UL_y, cost_unif, cost_non_unif
        UL_X = np.delete(UL_X, x_star, 0)
        UL_y = np.delete(UL_y, x_star, 0)
        cost_unif = np.delete(cost_unif, x_star, 0)
        cost_non_unif = np.delete(cost_non_unif, x_star, 0)

    return accuracy, total_cost[1:], oracle


def PlotErrorCost(accuracy, cost, cost_ratio):
    '''
    Utility function for plotting the error-cost curves for all three modes at a certain cost_ratio
    '''
    error = []
    for acc in accuracy:
        err = 1 - np.array(acc)
        error.append(err)

    plt.clf()
    plt.plot(cost[0], error[0], 'b', label='proactive')
    plt.plot(cost[1], error[1], 'r', label='uniform')
    plt.plot(cost[2], error[2], 'g', label='random')
    plt.legend()
    plt.xlabel('Total cost')
    plt.ylabel('Classification Error')
    plt.savefig('./proactive_learning_{:f}.png'.format(cost_ratio))
    return



if __name__ == '__main__':
    # load the data
    data = pd.read_csv('./Data.csv')
    class_num = np.unique(data['Label']).shape[0]
    # Encode labels with value between 0 and n_classes-1
    le = preprocessing.LabelEncoder()
    data['Label'] = le.fit_transform(data['Label'])

    # randomly split the data into a train set and test set
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    # # pre-compute the cost of variable cost oracle
    # cost_non_unif = VariableCostOracle(train_data)

    # load the cost
    cost_non_unif = np.load('./cost.npy')

    budget = 30 # total budget

    # ratio between the mean cost of variable cost oracle and uniform cost oracle
    cost_ratios = [5, 1.1, 0.5]

    for cost_ratio in cost_ratios:
        accuracy = []
        cost = []
        oracle = []
        for mode in ['proactive', 'uniform', 'random']:
            # proactive learning
            accuracy_mode, cost_mode, oracle_mode = ProactiveLearning(train_data, test_data, budget, cost_non_unif, cost_ratio, class_num, mode)
            accuracy.append(accuracy_mode)
            cost.append(cost_mode)
            oracle.append(oracle_mode)

            print("Number of Queries: ", mode)
            print([len(list(group)) for key, group in groupby(sorted(oracle_mode))])
        # plot classification error vs cost
        PlotErrorCost(accuracy, cost, cost_ratio)
