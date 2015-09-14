import numpy as np
from matplotlib import pyplot as plt


def random_spanning_tree(mat, mst_mat=None, num_itrs=100000):
    ''' Generate random spanning trees and find sums of their edges to bootstrap p-value of
    the mimimum spanning tree.

    INPUT
    mat : ndarray
       adjacency matrix of whole network
    mst_mat : ndarray
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
    if mst_mat:
        mst_sum = np.sum(mat[mst_mat > 0.])
        print len(rst_sums[rst_sums > mst_sum]) / float(num_itrs)
    return rst_sums


