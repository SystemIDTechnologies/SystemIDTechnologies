"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA


def observerKalmanIdentificationAlgorithm(input_signal, output_signal):

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension
    output_dimension = output_signal.dimension
    number_steps = output_signal.number_steps

    # Get value of p that maximizes r(p+1) <= l: as a result n/m <= p <= l/r-1. Here n = 12.
    p = max(int(12 / output_dimension), int(number_steps / input_dimension - 1))

    # Build matrix U
    U = np.zeros([input_dimension * (p+1), number_steps])
    for i in range(0, p+1):
        U[i * input_dimension:(i + 1) * input_dimension, i:number_steps] = u[:, 0:number_steps - i]

    # Get Y
    Y = np.matmul(y, LA.pinv(U))

    # Get Markov parameters
    markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(p):
        markov_parameters.append(Y[:, i * input_dimension + input_dimension:(i + 1) * input_dimension + input_dimension])

    return markov_parameters
