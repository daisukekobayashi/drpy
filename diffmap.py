#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy
import scipy, scipy.linalg

from l2_distance import *


def diffmap(src, dim = 2, t = 1.0, sigma = 1.0, verbose = True, debug = False):

    start = time.time()
    if verbose:
        print("diffusion map running on %d points in %d dimensions" % src.shape)

    # Compute Gaussian kernel matrix
    if verbose:
        print("Compute Markov forward transition probability matrix with %f timesteps..." % \
              t)
    sumsrc = numpy.sum(src ** 2, 1)
    K = numpy.exp( -(sumsrc.reshape(sumsrc.shape[0], 1) + (sumsrc - 2 * numpy.dot(src, src.T)) / \
                   (2 * sigma ** 2)))

    # Compute Markov probability matrix with t timesteps
    p = numpy.sum(K, 0).reshape(K.shape[1], 1)
    K = K / (numpy.dot(p, p.T) ** t)
    p = numpy.sqrt(numpy.sum(K, 0)).reshape(K.shape[1], 1)
    K = K / numpy.dot(p, p.T)

    # Perform SVD
    K[numpy.isinf(K)] = 0
    K[numpy.isnan(K)] = 0

    if verbose:
        print("Perform eigen decomposition...")
    U, S, V = scipy.linalg.svd(K)

    cpU = numpy.array([])
    if debug:
        cpU = U.copy()
    U = U / U[:, 0].reshape(U.shape[0], 1)

    end = time.time()
    if verbose:
        print("Elapsed time... %f" % (end - start))

    if debug:
        debuginfo = {"markov_prob" : K, "U" : cpU, "S" : S, "V" : V }
        return (U[:, 1:dim + 1], debuginfo)
    else:
        return U[:, 1:dim + 1]

