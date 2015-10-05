import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

import utils

def plot_tree(mst_mat, plot_nodes=None, node_labels=None, **kwargs):
    ''' Draws the tree graph of the  minimum spanning tree.
    
    INPUT
    mst_mat : ndarray
       adjacency matrix of mimimum spanning tree
    node_labels : dict
       keys = node number; values = node label (e.g. name of brain region)
    '''
    plt.figure(figsize=(20,12))
    # ensure that matrix is not symmetric
    mst_mat = utils.format_adjacency_matrix(mst_mat, symmetric=False)
    # generate graph
    mst_graph = nx.from_numpy_matrix(mst_mat)
    # plot tree
    pos = nx.graphviz_layout(mst_graph, prog='twopi',args='')
    edges = nx.draw_networkx_edges(mst_graph, pos)
    if plot_nodes:
        nodes = nx.draw_networkx_nodes(mst_graph, pos, **kwargs)
        plt.colorbar(nodes)
    if node_labels:
         nx.draw_networkx_labels(mst_graph, pos, node_labels, font_weight='bold');
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_spanning_tree_distribution(mat, rst_sums, mst, bins=25, \
                                        title='Spanning Tree Distribution', \
                                        ylim=20000, savef=None):
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
    plt.figure(figsize=(8,6))
    rst_distr = plt.hist(rst_sums, bins=bins);
    mst_line = plt.plot([np.sum(mat[mst > 0.]), np.sum(mat[mst > 0.])], [0, ylim], 'r-')
    plt.title(title, fontsize=28);
    plt.xlabel('Sum of Edges Along Random Spanning Tree', fontsize=16);
    if savef:
        plt.savefig(savef)
    plt.show()
