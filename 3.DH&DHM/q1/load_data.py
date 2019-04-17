import numpy as np 
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def load_data(dtype):
    """ Loads "serum" and "urine" data. Computes linkage from hierarchical clustering.

    :param dtype: str "serum" or "urine"

    :returns X: data matrix 1000x25
    :returns y: true labels 1000x1
    :returns T: 3 element tree
        T[0] = linkage matrix from hierarchical clustering.  See https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
               for details. If you are unfamiliar with hierarchical clustering using scipy, the following is another helpful resource (We won't use dendrograms
               here, but he gives a nice explanation of how to interpret the linkage matrix):
               https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/ 

        T[1] = An array denoting the size of each subtree rooted at node i, where i indexes the array.  
               ie. The number of all children + grandchildren + ... + the node itself

        T[2] = dict where keys are nodes and values are the node's parent
        """

    dtype = dtype.lower()

    if dtype == "serum":
        X,y = make_blobs(1000,25,centers=2,cluster_std=13,random_state=0)
    elif dtype == "urine":
        X,y = make_blobs(1000,25,centers=2,cluster_std=23,random_state=0)
    else:
        print("Specify the data type as either 'serum' or 'urine' ")
        return 0

    n_samples = len(X)
    Z = linkage(X,method='ward')
    link = Z[:,:2].astype(int)
    subtree_sizes = np.zeros(link[-1,-1]+2)
    subtree_sizes[:n_samples] = 1
    parent = {}
    parent[2*(n_samples-1)] = 0 #set root node as 0
    for i in range(len(link)):
        left = link[i,0]
        right = link[i,1]
        current = i + n_samples
        subtree_sizes[current] = subtree_sizes[left] + subtree_sizes[right] 
        parent[left] = current
        parent[right] = current

    T = [link,subtree_sizes,parent]

    return X, y, T 











