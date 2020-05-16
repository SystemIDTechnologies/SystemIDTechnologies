"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np


def getObservabilityMatrix(A, C, p, tk, dt):

    (state_dimension, _) = A(tk).shape
    (output_dimension, _) = C(tk).shape

    O = np.zeros([p * output_dimension, state_dimension])

    O[0:output_dimension, :] = C(tk)

    if p <= 0:
        return np.zeros([p * output_dimension, state_dimension])
    if p == 1:
        O[0:output_dimension, :] = C(tk)
        return O
    if p > 1:
        O[0:output_dimension, :] = C(tk)
        for j in range(1, p):
            temp = A(tk)
            for i in range(0, j - 1):
                temp = np.matmul(A(tk + (i + 1) * dt), temp)
            O[j * output_dimension:(j + 1) * output_dimension, :] = np.matmul(C(tk + j * dt), temp)
        return O
