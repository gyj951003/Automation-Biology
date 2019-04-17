import numpy as np
from scipy.cluster.hierarchy import linkage
from scipy import stats
import matplotlib.pyplot as plt
from load_data import load_data
from update_empirical import update_empirical
from best_pruning_and_labeling import best_pruning_and_labeling
from assign_labels import assign_labels
from get_leaves import get_leaves
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import random

def compute_error(L,labels):
    """Compute the error

    :param L: labeling of leaf nodes
    :param labels: true labels of each node

    :returns error: error of predictions"""

    wrong = 0
    for i in range(len(labels)):
        if L[i] != labels[i]:
            wrong += 1
    error = wrong/len(labels)
    return error

def select_case_1(data,labels,T,budget,batch_size):
    """DH algorithm where we choose P proportional to the size of subtree rooted at each node

    :param data: Data matrix 1000x25
    :param labels: true labels
    :param T: 3 element tree
        T[0] = linkage matrix from hierarchical clustering.  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
               for details. If you are unfamiliar with hierarchical clustering using scipy, the following is another helpful resource (We won't use dendrograms
               here, but he gives a nice explanation of how to interpret the linkage matrix):
               https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

        T[1] = An array denoting the size of each subtree rooted at node i, where i indexes the array.
               ie. The number of all children + grandchildren + ... + the node itself

        T[2] = dict where keys are nodes and values are the node's parent
    :param budget: Number of iterations to make
    :param batch_size: Number of queries per iteration"""

    n_nodes = len(T[1]) #total nodes in T
    n_samples = len(data) #total samples in data
    L = np.zeros(n_nodes) #majority label
    p1 = np.zeros(n_nodes) #empirical label frequency
    n = np.zeros(n_nodes) #number of points sampled from each node
    error = []#np.zeros(n_samples) #error at each round
    root = n_nodes-1 #corresponds to index of root
    P = np.array([root])
    L[root] = 1

    for i in range(budget):
        v_selected = np.array([])

        for b in range(batch_size):
            #TODO: select a node from P proportional to the size of subtree rooted at each node
            #wv = (number of leaves of Tv)/n.
            w = []
            for j in range(len(P)):
                num_leaves = len(get_leaves([], P[j], T, n_samples))
                w.append(num_leaves/n_samples)

            #print("weights:", w)

            v = random.choices(population = range(len(P)), weights = w, k=1)
            v = P[v[0]]
            #print("Selected internal node:", v)

            ##TODO: pick a random leaf node from subtree Tv and query its label
            z = random.choice(get_leaves([], v, T, n_samples))
            #print("Selected to query:", z)
            l = labels[z]

            #TODO: update empirical counts and probabilities for all nodes u on path from z to v
            z = np.array([z])
            n, p1 = update_empirical(n,p1,v,z,l,T)

            v_selected = np.append(v_selected, v)
            v_selected = v_selected.astype(int)

        #TODO: update admissible A and compute scores; find best pruning and labeling
        P_best, L[v] = best_pruning_and_labeling(n,p1,v_selected,T,n_samples)
        #print("best Pruning:", P_best)
        #TODO: update pruning P and labeling L
        P = np.delete(P, np.argwhere(P==v))
        P = np.union1d(P, P_best)
        #print("Updated Pruning:", P)

        #TODO: temporarily assign labels to every leaf and compute error
        L = assign_labels(L,v_selected,v_selected,T,n_samples)
        e = compute_error(L,labels)
        error.append(e)

        if (i % 100 == 0):
            print(e)

    for v in P:
        #TODO: assign labels to all nodes under the current pruning
        L = assign_labels(L,v,v,T,n_samples)

    return L, np.array(error)

def select_case_2(data,labels,T,budget,batch_size):
    """DH algorithm where we choose P by biasing towards choosing nodes in areas where the observed labels are less pure

    :param data: Data matrix 1000x25
    :param labels: true labels
    :param T: 3 element tree
        T[0] = linkage matrix from hierarchical clustering.  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
               for details. If you are unfamiliar with hierarchical clustering using scipy, the following is another helpful resource (We won't use dendrograms
               here, but he gives a nice explanation of how to interpret the linkage matrix):
               https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

        T[1] = An array denoting the size of each subtree rooted at node i, where i indexes the array.
               ie. The number of all children + grandchildren + ... + the node itself

        T[2] = dict where keys are nodes and values are the node's parent
    :param budget: Number of iterations to make
    :param batch_size: Number of queries per iteration"""

    n_nodes = len(T[1]) #total nodes in T
    n_samples = len(data) #total samples in data
    L = np.zeros(n_nodes) #majority label
    p1 = np.zeros(n_nodes) #empirical label frequency
    n = np.zeros(n_nodes) #number of points sampled from each node
    error = []#np.zeros(n_samples) #error at each round
    root = n_nodes-1 #corresponds to index of root
    P = np.array([root])
    L[root] = 1

    for i in range(budget):
        v_selected = np.array([])

        for b in range(batch_size):
            #TODO: select a node from P biasing towards choosing nodes in areas where the observed labels are less pure

            w = np.array([])
            za = 0.95

            for j in range(len(P)):
                leaves = get_leaves([], P[j], T, n_samples)
                num_leaves = len(leaves)
                p_v = max(p1[P[j]], 1-p1[P[j]]) # majority label frequency
                p_up =  p_v + za * np.sqrt(p_v * (1-p_v)/num_leaves)
                wv = num_leaves/n_samples

                w = np.append(w, wv * (1.0 - p_up))

            if (np.sum(w) == 0):
                w = w + 1.0/len(w)
            else:
                w = w / np.sum(w)
            #print("weights:", w)

            v = random.choices(population = range(len(P)), weights = w, k=1)
            v = P[v[0]]
            #print("Selected internal node:", v)

            #TODO: pick a random leaf node from subtree Tv and query its label
            z = random.choice(get_leaves([], v, T, n_samples))
            #print("Selected to query:", z)
            l = labels[z]

            #TODO: update empirical counts and probabilities for all nodes u on path from z to v
            z = np.array([z])
            n, p1 = update_empirical(n,p1,v,z,l,T)

            v_selected = np.append(v_selected, v)
            v_selected = v_selected.astype(int)

        #TODO: update admissible A and compute scores; find best pruning and labeling
        P_best, L[v] = best_pruning_and_labeling(n,p1,v_selected,T,n_samples)
        #print("best Pruning:", P_best)
        #TODO: update pruning P and labeling L
        P = np.delete(P, np.argwhere(P==v))
        P = np.union1d(P, P_best)
        #print("Updated Pruning:", P)

        #TODO: temporarily assign labels to every leaf and compute error
        L = assign_labels(L,v_selected,v_selected,T,n_samples)
        e = compute_error(L,labels)
        error.append(e)

        if (i % 100 == 0):
            print(e)

    for v in P:
        #TODO: assign labels to all nodes under the current pruning
        L = assign_labels(L,v,v,T,n_samples)

    return L, np.array(error)
