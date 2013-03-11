#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import scipy, scipy.linalg


def mgs(src):

    # Perform Gram-Schmidt orthogonalization
    m, n = src.shape
    Q = src.copy()
    R = numpy.zeros((n, n))
    for i in xrange(n):
        R[i, i] = scipy.linalg.norm(Q[:, i], 2)
        Q[:, i] = Q[:, i] / R[i, i]
        if i < n - 1:
            for j in xrange(i + 1, n):
                R[i, j] = numpy.dot(Q[:, i], Q[:, j])
                Q[:, j] = Q[:, j] - R[i, j] * Q[:, i]

    return (Q, R)
