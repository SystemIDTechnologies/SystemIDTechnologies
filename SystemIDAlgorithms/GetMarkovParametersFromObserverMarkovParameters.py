"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np


def getMarkovParametersFromObserverMarkovParameters(observer_markov_parameters):

    # Dimensions
    output_dimension, input_dimension = observer_markov_parameters[0].shape

    # Get D
    markov_parameters = [observer_markov_parameters[0]]

    # Number of steps
    number_steps = len(observer_markov_parameters)

    # Extract Yk1 and Yk2
    Yk1 = []
    Yk2 = []
    for i in range(1, number_steps):
        Yk1.append(observer_markov_parameters[i][:, 0:input_dimension])
        Yk2.append(-observer_markov_parameters[i][:, input_dimension:output_dimension + input_dimension])

    # Get Yk
    for i in range(1, number_steps):
        Yk = Yk1[i - 1]
        for j in range(1, i + 1):
            Yk = Yk - np.matmul(Yk2[j - 1], markov_parameters[i - j])
        markov_parameters.append(Yk)

    return markov_parameters
