"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np
from scipy.linalg import fractional_matrix_power as matpow


def getInitialConditionResponseMarkovParameters(A, C, x0, number_steps):

    markov_parameters = [np.matmul(C(0), x0)]

    for i in range(number_steps - 1):
        markov_parameters.append(np.matmul(C(0), np.matmul(matpow(A(0), i), x0)))

    return markov_parameters
