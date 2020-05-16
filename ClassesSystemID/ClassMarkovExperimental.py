"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


from SystemIDAlgorithms.GetMarkovParametersFull import getMarkovParametersFull
from SystemIDAlgorithms.GetObserverMarkovParametersFull import getObserverMarkovParametersFull
from SystemIDAlgorithms.GetMarkovParametersFromObserverMarkovParameters import getMarkovParametersFromObserverMarkovParameters
from SystemIDAlgorithms.GetMarkovParametersFull_Frequency import getMarkovParametersFull_Frequency


class MarkovExperimental:
    def __init__(self, input_signal, output_signal):
        self.markov_parameters_full = getMarkovParametersFull(input_signal, output_signal)
        self.observer_markov_parameters_full, self.V = getObserverMarkovParametersFull(input_signal, output_signal)
        self.markov_parameters_from_observer_full = getMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters_full)


class MarkovExperimental_Frequency:
    def __init__(self, input_signal, output_signal):
        self.markov_parameters_full = getMarkovParametersFull_Frequency(input_signal, output_signal)
