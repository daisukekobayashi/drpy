#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
from numpy import sum, dot, ones, eye, isnan 

def l2_distance(a, b):
    d = numpy.lib.scimath.sqrt(sum(a ** 2, axis = 1).reshape(a.shape[0], 1) + \
                               sum(b.T ** 2, axis = 0) - 2 * dot(a, b.T))
    d = numpy.real(d)
    return d * (ones(d.shape[0]) - eye(d.shape[0]))
