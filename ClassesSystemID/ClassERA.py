"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 1.0.0
Date: August 2020
Python: 3.7.7
"""


from SystemIDAlgorithms.EigenSystemRealizationAlgorithm import eigenSystemRealizationAlgorithm
from SystemIDAlgorithms.EigenSystemRealizationAlgorithmFromInitialConditionResponse import eigenSystemRealizationAlgorithmFromInitialConditionResponse


class ERA:
    def __init__(self, markov_parameters, state_dimension):
        self.A, self.B, self.C, self.D, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt = eigenSystemRealizationAlgorithm(markov_parameters, state_dimension)


class ERANucNorm:
    def __init__(self, markov_parameters, state_dimension):
        self.A, self.B, self.C, self.D, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt = eRAtestNucNormOpti(markov_parameters, state_dimension)


class ERAFromInitialConditionResponse:
    def __init__(self, output_signals, state_dimension, input_dimension):
        self.A, self.B, self.C, self.D, self.x0, self.H0, self.H1, self.R, self.Sigma, self.St, self.Rn, self.Sigman, self.Snt = eigenSystemRealizationAlgorithmFromInitialConditionResponse(output_signals, state_dimension, input_dimension)
