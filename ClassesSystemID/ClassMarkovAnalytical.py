"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np

from SystemIDAlgorithms.GetMarkovParameters import getMarkovParameters
from SystemIDAlgorithms.GetObserverMarkovParameters import getObserverMarkovParameters


class MarkovAnalytical:
    def __init__(self, A, B, C, D, number_steps, **kwargs):
        self.markov_parameters = getMarkovParameters(A, B, C, D, number_steps)
        self.G = kwargs.get('G', np.array([[None]]))
        if np.max(self.G):
            self.observer_markov_parameters = getObserverMarkovParameters(A, B, C, D, self.G, number_steps)
