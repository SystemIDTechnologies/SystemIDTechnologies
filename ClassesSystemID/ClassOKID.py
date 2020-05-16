"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


from SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithm import observerKalmanIdentificationAlgorithm
from SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithmFull import observerKalmanIdentificationAlgorithmFull
from SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithmObserver import observerKalmanIdentificationAlgorithmObserver
from SystemIDAlgorithms.ObserverKalmanIdentificationAlgorithmObserverFull import observerKalmanIdentificationAlgorithmObserverFull
from SystemIDAlgorithms.GetMarkovParametersFromObserverMarkovParameters import getMarkovParametersFromObserverMarkovParameters


class OKIDObserver:
    def __init__(self, input_signal, output_signal):
        self.observer_markov_parameters = observerKalmanIdentificationAlgorithmObserver(input_signal, output_signal)
        self.markov_parameters = getMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters)


class OKIDObserverFull:
    def __init__(self, input_signal, output_signal):
        self.observer_markov_parameters = observerKalmanIdentificationAlgorithmObserverFull(input_signal, output_signal)
        self.markov_parameters = getMarkovParametersFromObserverMarkovParameters(self.observer_markov_parameters)


class OKID:
    def __init__(self, input_signal, output_signal):
        self.markov_parameters = observerKalmanIdentificationAlgorithm(input_signal, output_signal)


class OKIDFull:
    def __init__(self, input_signal, output_signal):
        self.markov_parameters = observerKalmanIdentificationAlgorithmFull(input_signal, output_signal)
