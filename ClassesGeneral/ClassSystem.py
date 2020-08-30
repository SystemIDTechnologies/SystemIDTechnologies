"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 1.0.0
Date: August 2020
Python: 3.7.7
"""


class System:
    def __init__(self, frequency, state_dimension, input_dimension, output_dimension, initial_states, name):
        self.frequency = frequency
        self.dt = 1 / frequency
        self.state_dimension = state_dimension
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.initial_states = initial_states
        self.x0 = self.initial_states[0][0]
        self.name = name
        self.system_type = 'general'


class LinearSystem(System):
    def __init__(self, frequency, state_dimension, input_dimension, output_dimension, initial_states, name, A, B, C, D):
        super().__init__(frequency, state_dimension, input_dimension, output_dimension, initial_states, name)
        self.system_type = 'linear'
        self.A = A
        self.B = B
        self.C = C
        self.D = D
