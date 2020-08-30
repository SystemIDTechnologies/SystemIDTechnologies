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


from SystemIDAlgorithms.GetObservabilityMatrix import getObservabilityMatrix
from SystemIDAlgorithms.GetDeltaMatrix import getDeltaMatrix

def identificationInitialCondition(input_signal, output_signal, A, B, C, D, tk):

    # Sizes
    output_dimension, input_dimension = D(tk).shape

    # Number of steps and dt
    number_steps = input_signal.number_steps
    p = max(int(12 / output_dimension), int(number_steps / (input_dimension + output_dimension) - input_dimension))
    #number_steps = min(number_steps, p)
    dt = input_signal.dt

    # Data
    u = input_signal.data[:, 0:number_steps]
    y = output_signal.data[:, 0:number_steps]

    # Build U and Y
    U = u.T.reshape(1, number_steps * input_dimension).reshape(number_steps * input_dimension, 1)
    Y = y.T.reshape(1, number_steps * output_dimension).reshape(number_steps * output_dimension, 1)

    # Get the Observability matrix
    O = getObservabilityMatrix(A, C, number_steps, tk, dt)

    # Get the Delta Matrix
    Delta = getDeltaMatrix(A, B, C, D, tk, dt, number_steps)

    # Get initial condition
    xtk = np.matmul(LA.pinv(O), Y - np.matmul(Delta, U))
    print('Error IC: ', LA.norm(Y - np.matmul(O, xtk) - np.matmul(Delta, U)))

    return xtk

