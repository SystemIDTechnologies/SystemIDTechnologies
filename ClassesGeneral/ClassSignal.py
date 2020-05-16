"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np

from SystemIDAlgorithms.Propagation import propagation


class Signal:
    def __init__(self, total_time, frequency, dimension, name, **kwargs):
        self.total_time = total_time
        self.frequency = frequency
        self.dt = 1/frequency
        self.number_steps = int(self.total_time * self.frequency) + 1
        self.dimension = dimension
        self.name = name

        self.mean = kwargs.get('mean', np.array([[None]]))
        self.standard_deviation = kwargs.get('standard_deviation', np.array([[None]]))
        self.magnitude_impulse = kwargs.get('magnitude_impulse', np.array([[None]]))
        self.data = kwargs.get('data', np.array([[None]]))

        if np.max(self.mean) or np.max(self.standard_deviation):
            self.data = np.matmul(self.standard_deviation, np.random.randn(self.dimension, self.number_steps)) + self.mean[:,np.newaxis]
            self.signal_type = 'white_noise'
        elif np.max(self.magnitude_impulse):
            self.data = np.zeros([self.dimension, self.number_steps])
            self.data[:, 0] = self.magnitude_impulse[0]
            self.signal_type = 'impulse'
        elif np.max(self.data):
            self.signal_type = 'external'
        else:
            self.mean = np.zeros([self.dimension, 1])
            self.standard_deviation = np.zeros([self.dimension, self.dimension])
            self.magnitude_impulse = np.zeros([self.dimension, 1])
            self.data = np.zeros([self.dimension, self.number_steps])
            self.signal_type = 'zero'


class OutputSignal(Signal):
    def __init__(self, Signal, System, name):
        super().__init__(Signal.total_time, System.frequency, System.output_dimension, name, data=propagation(Signal, System)[0])
        self.state = propagation(Signal, System)[1]


def subtract2Signals(signal1, signal2):
    return Signal(signal1.total_time, signal1.frequency, signal1.dimension, 'Error: ' + signal1.name + ' - ' + signal2.name, data=signal1.data-signal2.data)


def add2Signals(signal1, signal2):
    return Signal(signal1.total_time, signal1.frequency, signal1.dimension, 'Sum: ' + signal1.name + ' + ' + signal2.name, data=signal1.data+signal2.data)