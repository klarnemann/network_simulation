import numpy as np
from matplotlib import pyplot as plt


def random_spanning_tree(mat, mst=None, num_itrs=100000):
    '''
    Generate random spanning trees and find sums of their edges to bootstrap p-value of
    the mimimum spanning tree.

    INPUT
    mat : ndarray
       adjacency matrix of whole network
    mst : ndarray
       adjacency matrix of  mimimum spanning tree
    num_itrs : int
       number of random samples to generate

    RETURNS
    rst_sums : ndarray
       array of sums of edges along random spanning trees
    '''
    # ensure that matrix is symmetric
    if not np.allclose(mat.T, mat):
        mat = np.triu(mat)
        mat = mat + mat.T
    # remove self-loops
    np.fill_diagonal(mat, 0.)
    # find sums along random spanning trees
    x, _ = np.shape(mat)
    rst_sums = []
    while num_itrs > 0:
        y = np.random.choice(80, 80)
        np.random.shuffle(x)
        rst_sums.append(np.sum(mat[x,y]))
        num_itrs -= 1
    np.array(rst_sums)
    # print p-value (bootstrapped)
    if mst:
        mst_sum = np.sum(mat[mst > 0.])
        print len(rst_sums[rst_sums > mst_sum]) / float(num_itrs)
    return rst_sums


def plot_spanning_tree_distribution(mat, rst_sums, mst, bins=25, \
                                        title='Spanning Tree Distribution', ylim=20000):
    '''
    Plots the distribution of the sums of edges along random spanning trees and as well as 
    the sum of edges along the minimum spanning tree.

    mat : ndarray
       adjacency matrix of whole network
    rst_sums : ndarray
       array of sums of edges along random spanning trees
    mst : ndarray
       adjacency matrix of  mimimum spanning tree
    '''
    rst_distr = plt.hist(rst_sums, bins=bins);
    mst_line = plt.plot([np.sum(mat[mst > 0.]), np.sum(mat[mst > 0.])], [0, ylim], 'r-')
    plt.title(title, fontsize=28);
    plt.xlabel('Sum of Edges Along Random Spanning Tree', fontsize=16);
    plt.legend(handles=[rst_distr, mst_line])
    plt.show()
