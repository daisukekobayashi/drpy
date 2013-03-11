#! /usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy
import scipy, scipy.linalg


def pca(src, dim = 2, verbose = True, debug = False):

    start = time.time()
    if verbose:
        print("PCA running on %d points in %d dimensions" % src.shape)

    # Calculate the deviations from the mean
    centered = src - src.mean(0)
    # Find the covariance matrix
    covm = numpy.dot(centered.T, centered) / centered.shape[0]
    # Find the eigenvectors and eigenvalues of the covariance matrix
    val, vec = scipy.linalg.eig(covm)
    # Rearrange the eigenvectors and eigenvalues
    index = numpy.real(val).argsort()
    index = index[::-1]

    # Convert the source data to z-scores
    sd = numpy.sqrt(numpy.diag(covm))
    z_scores = centered / sd

    end = time.time()

    if verbose:
        print("Elapsed time... %f" % (end - start))

    if debug:
        debuginfo = {"covariance_matrix" : covm, "z_scores" : z_scores, \
                     "eigval" : val, "eigvec" : vec}
        return (numpy.dot(z_scores, numpy.real(vec)[:,index[:dim]]), debuginfo)
    else:
        return numpy.dot(z_scores, numpy.real(vec)[:,index[:dim]])

