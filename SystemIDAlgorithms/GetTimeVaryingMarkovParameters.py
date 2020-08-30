"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 1.0.0
Date: August 2020
Python: 3.7.7
"""


import numpy as np


def getTimeVaryingMarkovParameters(A, B, C, D, tk1, tk2, number_steps):

    if tk1 == tk2:
        return D(tk1)
    elif number_steps == 1:
        return np.matmul(C(tk2), B(tk1))
    else:
        dt = (tk2-tk1) / number_steps
        Phi = B(tk1)
        for i in range(1, number_steps):
            Phi = np.matmul(A(tk1 + i*dt), Phi)
        return np.matmul(C(tk2), Phi)
