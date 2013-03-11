#! /usr/bin/env python
# -*- coding: utf-8 -*-

from numpy import *
from scipy import rand, randn

def example(data_name, num_data, noise = 0.05, num_labels = 7):
    if data_name == "swiss":
        data = zeros((num_data, 3))
        t = (3.0 * pi / 2.0) * (1.0 + 2.0 * rand(num_data))
        data[:, 0] = t * cos(t)
        data[:, 1] = 30.0 * rand(num_data)
        data[:, 2] = t * sin(t) + noise * randn(num_data)
        labels = floor(fmod(sqrt(data[:, 0] ** 2 + data[:, 2] ** 2), num_labels))
        return (data, labels)
