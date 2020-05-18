"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np
from SystemIDAlgorithms.GetTimeVaryingMarkovParameters import getTimeVaryingMarkovParameters


def getDeltaMatrix(A, B, C, D, tk, dt, number_steps):

    # Get dimensions
    output_dimension, input_dimension = D(tk).shape

    # Get Delta Matrix
    Delta = np.zeros([number_steps * output_dimension, number_steps * input_dimension])
    for i in range(number_steps):
        for j in range(i+1):
            Delta[i*output_dimension:(i+1)*output_dimension, j*input_dimension:(j+1)*input_dimension] = getTimeVaryingMarkovParameters(A, B, C, D, j*dt + tk, i*dt + tk, i-j)

    return Delta

