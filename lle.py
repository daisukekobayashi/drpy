#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy
import scipy, scipy.linalg, scipy.sparse, scipy.sparse.linalg

from l2_distance import *

def lle(src, dim = 2, k = 10, verbose = True, debug = False):

    start = time.time()
    if verbose:
        print("LLE running on %d points in %d dimensions" % src.shape)

    # STEP1: Compute pairwise distances & Find neighbors
    if verbose:
        print("Finding %d nearest neighbors..." % k)
    dist = l2_distance(src, src)
    cpdist = numpy.array([])
    if debug:
        cpdist = dist.copy()
    rn = xrange(dist.shape[0])
    dist[rn, rn] = numpy.inf

    neighborhood = dist.argsort(1)

    # STEP2: Solve for Reconstruction Weights
    tol = 0
    if k > dim:
        tol = 1e-3

    if verbose:
        print("Solving for reconstruction weights...")

    W = numpy.zeros((src.shape[0], k))
    for i in xrange(W.shape[0]):
        z = src[neighborhood[i, :k], :] - src[i]
        C = numpy.dot(z, z.T)
        C = C + numpy.eye(k) * tol * numpy.trace(C)
        W[i, :] = scipy.linalg.solve(C, numpy.ones((k, 1))).T
        W[i, :] = W[i, :] / W[i, :].sum()

    # STEP3: Compute Embedding from Eigenvects of Cost Matrix M = (1 - W).T (1 - W)
    # M = scipy.sparse.csc_matrix(numpy.eye(src.shape[0]))
    if verbose:
        print("Computing embedding...")
    M = numpy.eye(src.shape[0])
    for i in xrange(M.shape[0]):
        w = W[i, :]
        j = neighborhood[i, :k]
        M[i, j] = M[i, j] - w
        M[j, i] = M[j, i] - w
        for l in xrange(w.shape[0]):
            M[j[l], j] = M[j[l], j] + w[l] * w

    # Calculation of Embedding
    val, vec = scipy.linalg.eig(M)
    index = numpy.real(val).argsort()
    index = index[::-1]

    end = time.time()
    if verbose:
        print("Elapsed time... %f" % (end - start))

    if debug:
        debuginfo = {"distance_matrix" : cpdist, "reconst_weight" : W, \
                     "cost_matrix" : M, "eigval" : val, "eigvec" : vec}
        return (numpy.real(vec)[:, index[-(dim + 1):-1]] * numpy.sqrt(src.shape[0]), \
                debuginfo)
    else:
        return numpy.real(vec)[:, index[-(dim + 1):-1]] * numpy.sqrt(src.shape[0])

