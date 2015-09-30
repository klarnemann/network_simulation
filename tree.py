import utils
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt


def random_spanning_tree(mat, seed=None, mst_mat=None, num_itrs=100000):
    ''' Generate random spanning trees and find sums of their edges to bootstrap p-value of the mimimum spanning tree.

    Parameters
    ----------
    mat : ndarray
       adjacency matrix of whole network
    mst_mat : ndarray
       adjacency matrix of  mimimum spanning tree
    num_itrs : int
       number of random samples to generate

    Returns
    -------
    rst_sums : ndarray
       array of sums of edges along random spanning trees
    '''
    mat = utils.format_adjacency_matrix(mat)
    # find sums along random spanning trees
    x = range(np.shape(mat)[0])
    rst_sums = []
    while num_itrs > 0:
        np.random.shuffle(x)
        if seed:
            x.remove(seed)
            x = [seed] + x
        y = []
        for i in range(1,80):
              y.extend(np.random.choice(x[:i], 1))
        rst_sums.append(np.sum(mat[x[1:],y]))
        num_itrs -= 1
    np.array(rst_sums)
    # print p-value (bootstrapped)
    if mst_mat:
        mst_sum = np.sum(mat[mst_mat > 0.])
        print len(rst_sums[rst_sums > mst_sum]) / float(num_itrs)
    return rst_sums

def kruskals_minimum_spanning_tree(g):
    ''' Find minimum spanning tree based on distance edge attribute.

    Parameters
    ----------
    g: weighted undirected networkx graph

    Returns
    -------
    mst_g : weighted undirected networkx graph
       mimimum spanning tree graph
    mst_mat : ndarray
       minimum spanning tree adjacency matrix
    '''
    if not g.get_edge_data(0,1)['distance']:
        utils.add_distance_edge_attr(g)
    mst_g = nx.minimum_spanning_tree(g, weight='distance')
    mst_mat = nx.to_numpy_matrix(mst_g)
    return mst_g, mst_mat

def prims_minimum_spanning_tree(input_g, seed=None):
    """
    Prim's Algorithm : Finds order of nodes added to minimum spanning tree.
    
    Parameters
    ----------
    input_g : weighted networkx graph
    seed : int
       starting seed node of networkx graph
       if None, seed node randomly selected
        
    Returns
    -------
    mst_g : networkx graph
       minimum spanning tree graph
    curr_nodes : list
       order nodes added to the tree
    """
    if not seed:
        seed = int(np.random.choice(input_g.nodes(),1))
    if not input_g.get_edge_data(0,1)['distance']:
        utils.add_distance_edge_attr(input_g)
    g = input_g.copy()
    curr_nodes, curr_edges = [seed], []
    nodes = g.nodes()
    while len(curr_nodes) < len(nodes):
        curr_close, curr_dist, curr_edge = 1, np.inf, (0, 0)
        for node in curr_nodes:
            for edge in g.edges(node, data = True):
                if edge[1] in curr_nodes:
                    continue
                else:
                    if edge[2]['distance'] < curr_dist:
                        g.remove_edge(edge[0], edge[1])
                        curr_close, curr_dist, curr_edge = edge[1], edge[2]['distance'], edge
        curr_nodes.append(curr_close)
        curr_edges.append(curr_edge)
    mst_g = nx.Graph()
    mst_g.add_edges_from(curr_edges)
    return mst_g, curr_nodes

