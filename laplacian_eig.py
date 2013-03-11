#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy
import scipy, scipy.linalg

from l2_distance import *


def laplacian_eig(src, dim = 2, k = 10, heat = 1.0, verbose = True, debug = False):

    start = time.time()
    if verbose:
        print("Laplacian Eigenmap running on %d points in %d dimensions" % src.shape)

    if verbose:
        print("Constructing the adjacency graph...")
    dist = l2_distance(src, src)
    cpdist = dist.copy()

    graph = numpy.zeros(dist.shape)
    for i in xrange(dist.shape[0]):
        dist[i, i] = numpy.inf
        for j in xrange(k):
            idx = dist[i].argmin()
            graph[i, idx] = 1.0
            graph[idx, i] = graph[i, idx]
            dist[i, idx] = numpy.inf

    # Step 2: Choosing the weights
    if heat != 0:
        nz = numpy.nonzero(graph)
        graph[nz] = graph[nz] * numpy.exp(-cpdist[nz] / heat)

    # Laplacian matrix
    weight = numpy.diag(graph.sum(1))
    laplacian = weight - graph
    laplacian[numpy.isinf(laplacian)] = 0
    laplacian[numpy.isnan(laplacian)] = 0

    # Generalized Eigenvalue Decomposition
    if verbose:
        print("Constructing Eigenmaps...")
    val, vec = scipy.linalg.eig(laplacian, weight)
    index = numpy.real(val).argsort()

    end = time.time()
    if verbose:
        print("Elapsed time... %f" % (end - start))

    if debug:
        debuginfo = {"distance_matrix" : cpdist, "graph" : graph, \
                     "laplacian_graph" : laplacian, "eigval" : val, "eigvec" : vec}
        return (numpy.real(vec)[:, index[1:dim + 1]], debuginfo)
    else:
        return numpy.real(vec)[:, index[1:dim + 1]] 

