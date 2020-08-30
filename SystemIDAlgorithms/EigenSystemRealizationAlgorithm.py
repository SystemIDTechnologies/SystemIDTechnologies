"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 1.0.0
Date: August 2020
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power as matpow


def eigenSystemRealizationAlgorithm(markov_parameters, state_dimension):

    # Sizes
    p = int(np.floor((len(markov_parameters) - 1) / 2))
    if markov_parameters[0].shape == ():
        (output_dimension, input_dimension) = (1, 1)
    else:
        (output_dimension, input_dimension) = markov_parameters[0].shape

    # Hankel matrices H(0) and H(1)
    H0 = np.zeros([p * output_dimension, p * input_dimension])
    H1 = np.zeros([p * output_dimension, p * input_dimension])
    for i in range(p):
        for j in range(p):
            H0[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = markov_parameters[i + j + 1]
            H1[i * output_dimension:(i + 1) * output_dimension, j * input_dimension:(j + 1) * input_dimension] = markov_parameters[i + j + 2]

    # SVD H(0)
    print(H0.shape)
    (R, sigma, St) = LA.svd(H0, full_matrices=True)
    Sigma = np.diag(sigma)

    # Matrices Rn, Sn, Sigman
    Rn = R[:, 0:state_dimension]
    Snt = St[0:state_dimension, :]
    Sigman = Sigma[0:state_dimension, 0:state_dimension]

    # Identified matrices
    A_id = np.matmul(matpow(Sigman, -1/2),
                     np.matmul(np.transpose(Rn), np.matmul(H1, np.matmul(np.transpose(Snt), matpow(Sigman, -1/2)))))
    B_temp = np.matmul(matpow(Sigman, 1/2), Snt)
    B_id = B_temp[:, 0:input_dimension]
    C_temp = np.matmul(Rn, matpow(Sigman, 1/2))
    C_id = C_temp[0:output_dimension, :]
    D_id = markov_parameters[0]

    def A(tk):
        return A_id

    def B(tk):
        return B_id

    def C(tk):
        return C_id

    def D(tk):
        return D_id

    return A, B, C, D, H0, H1, R, Sigma, St, Rn, Sigman, Snt
