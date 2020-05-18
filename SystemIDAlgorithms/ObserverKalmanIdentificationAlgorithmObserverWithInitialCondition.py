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


def observerKalmanIdentificationAlgorithmObserverWithInitialCondition(input_signal, output_signal):

    # Get data from Signals
    y = output_signal.data
    u = input_signal.data

    # Get dimensions
    input_dimension = input_signal.dimension
    output_dimension = output_signal.dimension
    number_steps = output_signal.number_steps

    # Get value of p that maximizes (r+m)p + r <= l: as a result n/m <= p <= l/(r+m)-r. Here n = 12.
    p = max(int(12/output_dimension), int(number_steps/(input_dimension+output_dimension) - input_dimension))
    print('p = ', p)
    print('number_steps = ', number_steps)
    # Build matrix U
    U = np.zeros([(input_dimension + output_dimension) * p + input_dimension, number_steps - p])
    print('Shape of U = ', U.shape)
    U[0 * input_dimension:(0 + 1) * input_dimension, :] = u[:, p:number_steps]
    for i in range(0, p):
        U[i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension, :] = np.concatenate((u[:, p - 1 - i:number_steps - 1 - i], y[:, p - 1 - i:number_steps - 1 - i]), axis=0)


    # Get Y
    Y = np.matmul(y[:, p:number_steps], LA.pinv(U))
    print('p = ', p)
    print('Shape U', U.shape)
    print('Error OKID: ', LA.norm(y[:, p:number_steps]-np.matmul(Y, U)))

    # Get observer Markov parameters
    observer_markov_parameters = [Y[:, 0:input_dimension]]
    for i in range(p):
        observer_markov_parameters.append(Y[:, i * (input_dimension + output_dimension) + input_dimension:(i + 1) * (input_dimension + output_dimension) + input_dimension])

    return observer_markov_parameters, y, U
