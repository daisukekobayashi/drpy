#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy
import scipy, scipy.linalg

from l2_distance import *


def ltsa(src, dim = 2, k = 10, verbose = True, debug = False):

    start = time.time()
    if verbose:
        print("LTSA running on %d points in %d dimensions" % src.shape)

    #Compute pairwise distances & Find neighbors
    if verbose:
        print("Find %d nearest neighbors..."  % k)
    dist = l2_distance(src, src)
    cpdist = numpy.array([])
    if debug:
        cpdist = dist.copy()
    n = src.shape[0]
    #rn = xrange(n)
    #dist[rn, rn] = numpy.inf

    neighborhood = dist.argsort(1)

    # Compute local information matrix for all datapoints
    if verbose:
        print("Compute local information matrices for all datapoints...")
    gl = []
    for i in xrange(n):
        # Compute correlation matrix W
        li = neighborhood[i, :k]
        kt = li.shape[0]
        srci = src[li, :] - numpy.mean(src[li, :], 0)
        W = numpy.dot(srci, srci.T)
        W = (W + W.T) / 2.0

        # Compute local information by computing d largest eigenvectors of W
        Si, Vi = scipy.linalg.schur(W)
        index = (-numpy.diag(Si)).argsort()
        Vi = Vi[:, index[:dim]]

        # Store eigenvectors in G (Vi is the space with the maximum variance.
        # i.e. a good approximation of the tangent space at point srci)

        # The constant 1 / sqrt(kt) serves as a centering matrix
        Gi = numpy.hstack((numpy.ones((kt, 1)) / numpy.sqrt(kt), Vi))
        # Compute gl = I - Gi * Gi.T
        gl.append(numpy.eye(kt) - numpy.dot(Gi, Gi.T))

    # Construct sparse matrix G (= alignment matrix)
    if verbose:
        print("Construct alignment matrix...")
    G = numpy.eye(n)
    for i in xrange(n):
        li = neighborhood[i, :k]
        for j in xrange(len(li)):
            G[li[j], li] = G[li[j], li] + gl[i][j]
        G[i, i] = G[i, i] - 1

    G = (G + G.T) / 2.0
    G[numpy.isinf(G)] = 0
    G[numpy.isnan(G)] = 0

    # Perform eigenanalysis of matrix G
    if verbose:
        print('Perform eigenanalysis...')
    val, vec = scipy.linalg.eig(G)
    index = val.argsort()

    end = time.time()
    if verbose:
        print("Elapsed time... %f" % (end - start))

    if debug:
        debuginfo = {"distance_matrix" : cpdist, "alignment_matrix" : G, \
                     "eigval" : val, "eigvec" : vec}
        return (numpy.real(vec[:, index[1:dim + 1]]), debuginfo)
    else:
        return numpy.real(vec[:, index[1:dim + 1]])

