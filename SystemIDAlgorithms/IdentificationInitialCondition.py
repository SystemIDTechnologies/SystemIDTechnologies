"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


from SystemIDAlgorithms.GetObservabilityMatrix import getObservabilityMatrix

def identificationInitialCondition(input_signal, output_signal, A, B, C, D):

    # Sizes
    output_dimension, input_dimension = D(0).shape

    # Number of steps and dt
    number_steps = input_signal.number_steps
    dt = input_signal.dt

    # Get the Observability matrix
    O = getObservabilityMatrix(A, C, number_steps, 0, dt)

    # Build the Delta matrix
    Delta =