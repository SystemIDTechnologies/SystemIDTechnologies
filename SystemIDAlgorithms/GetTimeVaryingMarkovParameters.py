"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np
from scipy.linalg import fractional_matrix_power as matpow


def getTimeVaryingMarkovParameters(A, B, C, D, number_steps):