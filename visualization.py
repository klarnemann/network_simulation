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
