#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy
import scipy, scipy.linalg

from l2_distance import *
from mgs import *


def hlle(src, dim = 2, k = 10, verbose = True, debug = False):

    start = time.time()
    if verbose:
        print("Hessian LLE running on %d points in %d dimensions" % src.shape)

    # STEP1: Compute pairwise distances & Find neighbors
    dist = l2_distance(src, src)
    cpdist = numpy.array([])
    if debug:
        cpdist = dist.copy()
    rn = xrange(dist.shape[0])
    dist[rn, rn] = numpy.inf

    neighborhood = dist.argsort(1)

    # Size of original data
    n = src.shape[0]

    # Extra term count for quadratic form
    dp = dim * (dim + 1) / 2
    W = numpy.zeros((n * dp, n))

    # For all datapoints
    print("Building Hessian estimator for neighboring points...")
    for i in xrange(src.shape[0]):
        # Center datapoints by substracting their mean
        tmp_ind = neighborhood[i, :k]
        thissrc = src[tmp_ind, :]
        thissrc = (thissrc - numpy.mean(thissrc, 0)).T

        # Compute local coordinates (using SVD)
        U, D, Vpr = scipy.linalg.svd(thissrc)
        Vpr = Vpr.T
        V = Vpr[:, :dim]

        # Build Hessian estimator
        ct = 0
        Yi = numpy.zeros((V.shape[0], numpy.sum(xrange(1, dim + 1))))
        for mm in xrange(dim):
            startp = V[:, mm]
            for nn in xrange(len(xrange(mm, dim))):
                indles = xrange(mm, dim)
                Yi[:, ct + nn] = startp * V[:, indles[nn]]
            ct = ct + len(xrange(mm, dim))
        Yi = numpy.hstack((numpy.ones((k, 1)), V, Yi))

        # Gram-Schmidt orthogonalization
        Yt, Orig = mgs(Yi)
        Pii = Yt[:, dim + 1:].T

        # Double check weights sum to 1
        for j in xrange(dp):
            if numpy.sum(Pii[j, :]) > 0.0001:
                tpp = Pii[j, :] / numpy.sum(Pii[j, :])
            else:
                tpp = Pii[j, :]

            # Fill weight array
            W[i * dp + j, tmp_ind] = tpp

    if verbose:
        print("Computing HLLE embedding (eigenanalysis)...")

    G = numpy.dot(W.T, W)
    G[numpy.isnan(G)] = 0

    val, vec = scipy.linalg.eig(G)
    cpvec = numpy.array([])
    if debug:
        cpvec = vec.copy()

    index = numpy.real(val).argsort()
    vec = numpy.real(vec[:, index[1:dim + 1]])

    vec = vec[:, :dim] * numpy.sqrt(n)

    end = time.time()
    if verbose:
        print("Elapsed time... %f" % (end - start))

    if debug:
        debuginfo = {"distance_matrix" : cpdist, "weight_matrix" : W, \
                     "eigval" : val, "eigvec" : cpvec}
        return (vec, debuginfo)
    else:
        return vec
 
