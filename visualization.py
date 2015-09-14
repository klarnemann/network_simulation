import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def plot_tree(mst_mat, node_labels):
    ''' Draws the tree graph of the  minimum spanning tree.
    
    INPUT
    mst_mat : ndarray
       adjacency matrix of mimimum spanning tree
    node_labels : dict
       keys = node number; values = node label (e.g. name of brain region)
    '''
    # ensure that matrix is not symmetric
    if np.allclose(mst_mat.T, mst_mat):
        mst_mat = np.triu(mst_mat)
    # remove self-loops
    np.fill_diagonal(mst_mat, 0.)
    # generate graph
    mst_graph = nx.from_numpy_matrix(mst_mat)
    # plot tree
    pos = nx.graphviz_layout(mst_graph, prog='twopi',args='')
    nx.draw_networkx_labels(mst_graph, pos, node_labels, font_weight='bold');
    plt.show()


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
