"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np


def Markov_parameter_calculator_matrix(k, l, A, B, C):

    ## Get the dimensions
    (state_dim, input_dim) = B[0].shape
    (output_dim, _) = C[0].shape

    ## Get Markov parameter
    h = np.zeros([output_dim, input_dim])
    if l > k-1:
        h = np.zeros([output_dim, input_dim])
    if l == k-1:
        h = np.matmul(C[:, :, k], B[:, :, k-1])
    else:
        temp = np.matmul(A[:, :, l+1], B[:, :, l])
        for i in range(k-l-2):
            temp = np.matmul(A[:, :, l+2+i], temp)
        h = np.matmul(C[:, :, k], temp)
        
    return h
