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
from scipy.linalg import fractional_matrix_power as matpow


def eigenSystemRealizationAlgorithmFromInitialConditionResponse(output_signals, state_dimension):

    # Number of Signals
    number_signals = len(output_signals)

    # Number of steps
    number_steps = output_signals[0].number_steps

    # Dimensions
    output_dimension = output_signals[0].dimension

    # Building pseudo Markov parameters
    markov_parameters = []
    for i in range(number_steps):
        Yk = np.zeros([output_dimension, number_signals])
        for j in range(number_signals):
            Yk[:, j] = output_signals[j].data[:, i]
        markov_parameters.append(Yk)

    # Sizes
    # input_dimension = number_signals
    p = int(np.floor((len(markov_parameters) - 1) / 2))
    #p = 200
    if markov_parameters[0].shape == ():
        (output_dimension, input_dimension) = (1, 1)
    else:
        (output_dimension, input_dimension) = markov_parameters[0].shape

    # Hankel matrices H(0) and H(1)
    H0 = np.zeros([p * output_dimension, p * input_dimension])
    H1 = np.zeros([p * output_dimension, p * input_dimension])
    for i in range(p):
        for j in range(p):
            H0[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = markov_parameters[i + j]
            H1[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = markov_parameters[i + j + 1]

    # SVD H(0)
    (R, sigma, St) = LA.svd(H0, full_matrices=True)
    Sigma = np.diag(sigma)

    # Matrices Rn, Sn, Sigman
    Rn = R[:, 0:state_dimension]
    Snt = St[0:state_dimension, :]
    Sigman = Sigma[0:state_dimension, 0:state_dimension]

    # Identified matrices
    A_id = np.matmul(matpow(Sigman, -1 / 2),
                     np.matmul(np.transpose(Rn), np.matmul(H1, np.matmul(np.transpose(Snt), matpow(Sigman, -1 / 2)))))
    B_temp = np.matmul(matpow(Sigman, 1 / 2), Snt)
    x0_id = B_temp[:, 0:input_dimension]
    C_temp = np.matmul(Rn, matpow(Sigman, 1 / 2))
    C_id = C_temp[0:output_dimension, :]

    def A(tk):
        return A_id

    def C(tk):
        return C_id


    return A, C, x0_id, H0, H1, R, Sigma, St, Rn, Sigman, Snt
