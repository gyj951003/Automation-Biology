import numpy as np
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
from load_data import load_data
from update_empirical import update_empirical
from best_pruning_and_labeling import best_pruning_and_labeling
from assign_labels import assign_labels
from get_leaves import get_leaves
from dh import compute_error, select_case_1, select_case_2

def make_plot(x,y,part,dtype,fname):
    """Generates plot of fraction of incorrect labels vs number of iterations

    :param x: range(num_iterations)
    :param y: fraction incorrectly assigned labels
    :param part: which part of the homework
    :param dtype: serum or urine
    :param fname: output file name for plot
    """
    plt.plot(x,y)
    plt.xlabel("Number of iterations")
    plt.ylabel("Fraction of labels incorrect")
    plt.title("Part {0}: {1}".format(part,dtype))
    plt.savefig(fname)
    plt.clf()

def call_DH(part):
    """Main function to run all your code once complete.  After you complete
       select_case_1() and select_case_2(), this will run the DH algo for each
       dataset and generate the plots you will submit within your write-up.

       :param part: which part of the homework to run
    """

    part = part.lower()
    num_trials = 5

    if part == "b":
        print("Running part B")
        X, y, T = load_data("serum")
        l = np.zeros(len(X))
        for i in range(num_trials):
            print("Currently on iteration {}".format(i))
            L, error = select_case_1(X,y,T,len(X),1)
            l += error
        l /= num_trials
        make_plot(range(len(X)),l,"B","Serum random sampling","q1_part_B.png")

    elif part == "c":
        print("Running part C")
        X, y, T = load_data("urine")
        l = np.zeros(len(X))
        for i in range(num_trials):
            print("Currently on iteration {}".format(i))
            L, error = select_case_1(X,y,T,len(X),1)
            l += error
        l /= num_trials
        make_plot(range(len(X)),l,"C","Urine random sampling","q1_part_C.png")

    elif part == "d":
        print("Running part D")
        X, y, T = load_data("serum")
        l = np.zeros(len(X))
        for i in range(num_trials):
            print("Currently on iteration {}".format(i))
            L, error = select_case_2(X,y,T,len(X),1)
            l += error
        l /= num_trials
        make_plot(range(len(X)),l,"D","Serum active learning","q1_part_D.png")

    elif part == "e":
        print("Running part E")
        X, y, T = load_data("urine")
        l = np.zeros(len(X))
        for i in range(num_trials):
            print("Currently on iteration {}".format(i))
            L, error = select_case_2(X,y,T,len(X),1)
            l += error
        l /= num_trials
        make_plot(range(len(X)),l,"E","Urine active learning","q1_part_E.png")

    else:
        print("Incorrect part provided. Either 'b', 'c', 'd', or 'e' expected")


if __name__ == "__main__":

    for part in "bcde":
        call_DH(part)
