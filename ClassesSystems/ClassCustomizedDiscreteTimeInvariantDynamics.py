"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 1.0.0
Date: August 2020
Python: 3.7.7
"""


import numpy as np
from numpy.linalg import inv
from scipy.linalg import expm


class CustomizedDiscreteTimeInvariantDynamics:
    def __init__(self, A, B, C, D):
        self.state_dimension, _ = A.shape
        self.output_dimension, self.input_dimension = D.shape
        self.Ad = A
        self.Bd = B
        self.Cd = C
        self.Dd = D
        self.total_measurements = []
        self.units = []
        for i in range(self.output_dimension):
            self.total_measurements.append('Measurement {}'.format(i+1))
            self.units.append('SI')

    def A(self, tk):
        return self.Ad

    def B(self, tk):
        return self.Bd

    def C(self, tk):
        return self.Cd

    def D(self, tk):
        return self.Dd
