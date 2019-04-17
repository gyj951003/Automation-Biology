import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from sklearn.tree import DecisionTreeRegressor
import itertools

def call_oracle(x1_range,x2_range):
	'''Generate uniform sample over the input ranges and obtains the label for the sampled data point
	:param x1_range: tuple, indicating upper and lower bounds to sample from for feature x1
	:param x2_range: tuple, indicating upper and lower bounds to sample from for feature x1
	:returns sample: list, with single x1 and x2 coordinates
	:returns y_queried: float, label for the queried data point'''

	x1 = np.random.uniform(x1_range[0],x1_range[1])
	x2 = np.random.uniform(x2_range[0],x2_range[1])

	if x1 > 0.14 and x1 < 0.2 and x2 > 0.6 and x2 < 0.9:
		y_queried = np.random.normal(10,1)
	elif x1 > 0.61 and x1 < 0.86 and x2 > 0.14 and x2 < 0.36:
		y_queried = np.random.normal(20,1)
	else:
		y_queried = np.random.normal(1,1)

	sample = [x1,x2]

	return sample, y_queried

def test_data():
	'''Generate test data
	:returns X_test: list, 50000x2 x coordinates
	:returns y_test: list, labels corresponding to X_test '''

	X_test,y_test = [], []
	for i in range(50000):
		X_, y_ = call_oracle((0,1),(0,1))
		X_test.append(X_)
		y_test.append(y_)
	y_test = np.array(y_test) + np.random.normal(0,1,(len(y_test)))

	return X_test,y_test

def plot_samples(tree,all_samples,title,fname):
	'''Visualizes the queried samples at different steps of the RDP algo
	:param tree: dictionary, the tree
	:param all_samples: bool, if True plots all queried samples else only samples in the deepest leaves
	:param title: string, the title for the plot
	:param fname: string, output file name'''

	x_query = []
	y_query = []
	if all_samples:
		x_query = tree["0"]["X_queried"]
		y_query = tree["0"]["y_queried"]
	else:
		#find deepest leaves
		deepest = max([len(p) for p in tree])
		deep_leaves = [p for p in list(tree) if len(p) == deepest]

		#obtain samples from deepest leaves
		for l in list(deep_leaves):
			x_query.extend(tree[l]["X_queried"])
			y_query.extend(tree[l]["y_queried"])

	plt.scatter(np.array(x_query)[:,0],np.array(x_query)[:,1],c=y_query,s=0.01)
	plt.ylim((0,1))
	plt.xlim((0,1))
	plt.xlabel("X1")
	plt.ylabel("X2")
	plt.title(title)
	plt.savefig(fname)
	plt.clf()

def create_dyadic_tree(tree,current_node,max_depth):
	'''Creates RDP tree over the domain X1,X2 and associates samples from the domain to corresponding node in tree
	   :param tree: dict, tree as described in the prompt
	   :param current_node: string, name of current node.
	   :param max_depth: int, maximum desired depth of the tree
	   :returns tree: dict, tree updated with new nodes defined by performing an RDP'''

	if len(current_node) == max_depth + 1:
		tree[current_node]["leaf_status"] = True
		return tree

	X = tree[current_node]["X_queried"]
	X = np.asarray(X)
	Y = tree[current_node]["y_queried"]
	Y = np.asarray(Y)
	x1_middle = 0.5 * sum(tree[current_node]["x1_range"])
	x2_middle = 0.5 * sum(tree[current_node]["x2_range"])

	row02 = np.where(X[:,0] <= x1_middle)
	X02, Y02 = X[row02], Y[row02]
	row13 = np.where(X[:,0] > x1_middle)
	X13, Y13 = X[row13], Y[row13]

	row0 = np.where(X02[:,1] > x2_middle)
	X0, Y0 = X02[row0], Y02[row0]
	x1_range, x2_range = (np.min(X0[:, 0]), np.max(X0[:, 0])), (np.min(X0[:, 1]), np.max(X0[:, 1]))

	tree[current_node+"0"] = {"X_queried": X0.tolist(), "y_queried": Y0.tolist(),
		"x1_range": x1_range, "x2_range": x2_range, "leaf_status":False}
	create_dyadic_tree(tree, current_node+"0", max_depth)

	row2 = np.where(X02[:,1] <= x2_middle)
	X2, Y2 = X02[row2], Y02[row2]
	tree[current_node+"2"] = {"X_queried":X2.tolist(), "y_queried":Y2.tolist(),
		"x1_range":(np.min(X2[:, 0]), np.max(X2[:, 0])),
		"x2_range":(np.min(X2[:, 1]), np.max(X2[:, 1])),"leaf_status":False}
	create_dyadic_tree(tree, current_node+"2", max_depth)

	row1 = np.where(X13[:,1] > x2_middle)
	X1, Y1 = X13[row1], Y13[row1]
	tree[current_node+"1"] = {"X_queried":X1.tolist(), "y_queried":Y1.tolist(),
		"x1_range":(np.min(X1[:, 0]), np.max(X1[:, 0])),
		"x2_range":(np.min(X1[:, 1]), np.max(X1[:, 1])),"leaf_status":False}
	create_dyadic_tree(tree, current_node+"1", max_depth)

	row3 = np.where(X13[:,1] <= x2_middle)
	X3, Y3 = X13[row3], Y13[row3]
	tree[current_node+"3"] = {"X_queried":X3.tolist(), "y_queried":Y3.tolist(),
		"x1_range":(np.min(X3[:, 0]), np.max(X3[:, 0])),
		"x2_range":(np.min(X3[:, 1]), np.max(X3[:, 1])),"leaf_status":False}
	create_dyadic_tree(tree,current_node+"3",max_depth)

	return tree

def prune(tree):
	'''Prunes the tree according to the mean variance of samples at child nodes compared to their parents
	   Use a threshold difference of 2 when deciding whether or not to prune child leaf nodes
	   :param tree: dict, tree as described in the prompt
	   :return tree: dict, the pruned tree in the same format as the input tree  '''

	threshold = 2.0
	level = max([len(p) for p in tree])

	while (level > 2):
		level -= 1
		key_list = list(tree.keys())

		for k in key_list:
			if (len(k) == level):
				parent_y_var = np.var(np.asarray(tree[k]["y_queried"]))
				leaf_y_var = 0.0
				not_prune = False

				for i in ["0", "1", "2", "3"]:
					if (tree[k+i]["leaf_status"] == False):
						not_prune = True
						break
					leaf_y_var += np.var(np.asarray(tree[k+i]["y_queried"]))

				if (not_prune):
					continue

				leaf_y_var /= 4.0 # mean_var of leaves

				if (parent_y_var - leaf_y_var < threshold and
					parent_y_var - leaf_y_var > -1 * threshold ):
					# remove leaf_y
					for i in ["0", "1", "2", "3"]:
						tree.pop(k+i)

					tree[k]["leaf_status"] = True

	return tree

def refine(pruned_tree,n,max_depth):
	'''refine step of RDP algo.  Takes deepest leaves of pruned tree, resamples, and reprunes to identify
	   samples closest to each function border.
	   :param pruned_tree: dict, tree returned by prune() function
	   :param n: int, number of samples to obtain from the oracle
	   :param max_depth: int, max depth of rdp trees generated
	   :returns refined_rdps: dict, refined tree in same format as pruned_tree '''

	#get deepest leaves
	deepest_length = max([len(t) for t in pruned_tree])
	deep_leaves = [d for d in pruned_tree if len(d)==deepest_length]

	#sample from deepest leaves
	for i in range(n):
		choice = np.random.choice(deep_leaves)
		x_sample,y_sample = call_oracle(pruned_tree[choice]["x1_range"],pruned_tree[choice]["x2_range"])
		pruned_tree[choice]["X_queried"].append(x_sample)
		pruned_tree[choice]["y_queried"].append(y_sample)

	#create RDPs for each deep leaf
	max_depth = (max_depth-1)+len(deep_leaves[0])
	refined_rdps = {}
	for d in deep_leaves:
		init_subtree = {d:pruned_tree[d]}
		refined_subtree = create_dyadic_tree(init_subtree,d,max_depth)
		pruned_refined_subtree = prune(refined_subtree)
		refined_rdps.update(pruned_refined_subtree)

	return refined_rdps

def main():
	'''This function will call all parts of the HW in the correct order
	   It will generate relevant plots and descriptive statistics.
	   The tree is initialized here for you
	   After completing this function for part C, and implementing Create_dyadic_tree() and prune(),
	   calling main() should generate the necessary components asked for your homework submission.'''

	#obtain initial samples from oracle, set max depth
	print("Sampling from oracle")
	N = 500000
	n = int(N/2)
	X,y = [], []
	for i in range(n):
		X_q, y_q = call_oracle((0,1),(0,1))
		X.append(X_q)
		y.append(y_q)

	max_depth = int((1/3)*(np.log((float(n)/np.log(float(n))))))
	#add float coversio for stability in py2
	print("Max depth: {}".format(max_depth))

	#instantiate the tree by setting the root node
	print("Building tree")
	tree = {}
	tree["0"] = {"X_queried":X,"y_queried":y,"x1_range":(0,1),"x2_range":(0,1),"leaf_status":False}
	tree = create_dyadic_tree(tree,"0",max_depth)
	print("Total number of nodes: {}".format(len(tree)))

	#plot all initial samples contained in tree
	print("plotting all samples")
	plot_samples(tree,True,"All initial samples","initial_samples.png")


	#prune the tree
	print("pruning tree")
	pruned_tree = prune(tree)
	print("Number of nodes in pruned tree: {}".format(len(pruned_tree)))

	#plot samples in deepest leaves of pruned tree
	print("plotting post-pruning samples")
	plot_samples(pruned_tree,False,"Samples in deepest leaves post-pruning","post_pruning.png")

	#refine the pruned tree
	print("refining tree")
	refined_tree = refine(pruned_tree,n,max_depth)
	print("plotting post-refinement samples")
	plot_samples(refined_tree,False,"Samples after refinement step","post_refinement.png")

	#create test data
	X_test,y_test = test_data()

	#TODO: obtain deepest samples in refined_tree
	X_train, y_train = [], []
	#find deepest leaves
	deepest = max([len(p) for p in refined_tree])
	deep_leaves = [p for p in list(refined_tree) if len(p) == deepest]
	print("Number of deepest leaves: ", len(deep_leaves))
	#obtain samples from deepest leaves
	for l in list(deep_leaves):
		X_train.extend(refined_tree[l]["X_queried"])
		y_train.extend(refined_tree[l]["y_queried"])

	#TODO train DecisionTreeRegressor on deepest samples
	regressor = DecisionTreeRegressor()
	regressor.fit(X_train, y_train)
	r2 = regressor.score(X_test, y_test)
	print("The R2 value on test data is: ", r2)

if __name__ == '__main__':

	main()
