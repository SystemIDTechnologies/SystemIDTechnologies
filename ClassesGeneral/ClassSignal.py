"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np
import scipy.linalg as LA

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
        self.magnitude_peak = kwargs.get('magnitude_peak', np.array([[None]]))
        self.magnitude_sinusoid = kwargs.get('magnitude_sinusoid', np.array([[None]]))
        self.frequency_sinusoid = kwargs.get('frequency_sinusoid', np.array([[None]]))
        self.maximum_ramp = kwargs.get('maximum_ramp', np.array([[None]]))
        self.exponential_decay_rate = kwargs.get('exponential_decay_rate', np.array([[None]]))
        self.data = kwargs.get('data', np.array([[None]]))

        if np.max(self.maximum_ramp) or np.max(self.exponential_decay_rate):
            self.data = np.zeros([self.dimension, self.number_steps])
            for i in range(self.dimension):
                self.data[i, 0:int(self.number_steps/3)] = np.linspace(0, self.maximum_ramp[i], int(self.number_steps/3))
                self.data[i, int(self.number_steps/3):2*int(self.number_steps/3)] = LA.sqrtm(self.standard_deviation)[i, i]*np.random.randn(2*int(self.number_steps/3)-int(self.number_steps/3)) + self.maximum_ramp[i]
                self.data[i, 2*int(self.number_steps / 3):self.number_steps] = self.data[i, 2*int(self.number_steps/3)-1] * np.exp(self.exponential_decay_rate[i]*np.linspace(0, self.number_steps-2*int(self.number_steps / 3), self.number_steps-2*int(self.number_steps / 3)))
            self.signal_type = 'combination'
        elif np.max(self.mean) or np.max(self.standard_deviation):
            self.data = np.matmul(LA.sqrtm(self.standard_deviation), np.random.randn(self.dimension, self.number_steps)) + self.mean[:,np.newaxis]
            self.signal_type = 'white_noise'
        elif np.max(self.magnitude_impulse):
            self.data = np.zeros([self.dimension, self.number_steps])
            self.data[:, 0] = self.magnitude_impulse[0]
            self.signal_type = 'impulse'
        elif np.max(self.magnitude_peak):
            self.data = np.zeros([self.dimension, self.number_steps])
            for i in range(self.dimension):
                self.data[i, 0:int(self.number_steps/2)+1] = np.linspace(0, self.magnitude_peak[i], int(self.number_steps/2)+1)
                self.data[i, int(self.number_steps/2):self.number_steps] = np.linspace(self.magnitude_peak[i], 0, self.number_steps-int(self.number_steps/2))
            self.signal_type = 'triangle'
        elif np.max(self.magnitude_sinusoid) or np.max(self.frequency_sinusoid):
            self.data = np.zeros([self.dimension, self.number_steps])
            for i in range(self.dimension):
                self.data[i, :] = self.magnitude_sinusoid[i] * np.sin(2*np.pi*self.frequency_sinusoid[i]*np.linspace(0, self.total_time, self.number_steps))
            self.signal_type = 'sinusoid'
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