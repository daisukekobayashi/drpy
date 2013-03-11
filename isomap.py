#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
from numpy import *
import scipy, scipy.linalg
#import networkx as nx
from l2_distance import *
import pyublas, graph

def isomap(src, dim=2, k=10, verbose=True, debug=False):

    start = time.time()

    INF = 1000.0 * src.max() * src.shape[0]

    if verbose:
        print("Isomap running on %d points in %d dimensions" % src.shape)

    if verbose:
        print "Analysis graph structure of distance matrix..."
    dist = l2_distance(src, src)
    #G = nx.Graph()
    #for i in arange(dist.shape[0]):
    #        dist[i, i] = inf
    #        for j in arange(k):
    #            idx = dist[i].argmin()
    #            G.add_edge(i, idx, weight=dist[i, idx])
    #            dist[i, idx] = inf

    if verbose:
        print "Compute shortest path..."
    #gdist = zeros(dist.shape)
    #for row in range(gdist.shape[0]):
    #   for column in range(gdist.shape[1]):
    #       gdist[row][column] = nx.dijkstra_path_length(G, row, column)
    D = dist.copy()
    gdist = graph.geodesic_distance(dist, k, INF)

    mat_size = gdist.shape[0]
    G = gdist ** 2
    M = -0.5 * (G - dot(sum(G, axis=0).reshape(mat_size, 1), ones((1, mat_size))) \
                / mat_size - dot(ones((mat_size, 1)), sum(G, axis=0).reshape(1, mat_size)) \
                / mat_size + sum(G) / (mat_size ** 2))

    M[isinf(M)] = 0
    M[isnan(M)] = 0

    if verbose:
        print "Compute eigenvalue decomposition..."
    val, vec = scipy.linalg.eig(M)
    cpval = val.copy()
    cpvec = vec.copy()
    val, vec = real(val), real(vec)
    index = val.argsort()

    index = index[::-1]

    end = time.time()
    if verbose:
        print("Elapsed time... %f" % (end - start))

    if debug:
        debuginfo = {"distance_matrix" : D, "geodesic_distance" : gdist, \
                     "eigval" : cpval, "eigvec" : cpvec}
        return (vec[:, index[:dim]] * sqrt(val[index[:dim]]), debuginfo)
    else:
        return vec[:, index[:dim]] * sqrt(val[index[:dim]])

