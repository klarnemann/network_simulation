import numpy as np
import networkx as nx

def add_distance_edge_attr(g):
    ''' Add a distance attribute to graph edges.
    Parameters
    -----------
    g : weighted undirected networkx graph
    '''
    for edge in g.edges(data=True):
        edge[2]['distance'] = 1.0 - edge[2]['weight']

def format_adjacency_matrix(mat, symmetric=True):
    '''ensure that matrix is symmetric, no self-loops'''
    if symmetric:
        # ensure that matrix is symmetric
        if not np.allclose(mat.T, mat):
            mat = np.triu(mat)
        mat = mat + mat.T
    else:
        # ensure that matrix is not symmetric
        if np.allclose(mat.T, mat):
            np.triu(mat)
    # remove self-loops
    np.fill_diagonal(mat, 0.)
    return mat
