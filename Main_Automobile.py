"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 1.0.0
Date: August 2020
Python: 3.7.7
"""



import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt



from ClassesSystems.ClassAutomobileSystemDynamics import AutomobileSystemDynamics
from ClassesGeneral.ClassSystem import LinearSystem
from ClassesGeneral.ClassSignal import Signal, OutputSignal, subtract2Signals, add2Signals
from ClassesGeneral.ClassExperiments import Experiments
from ClassesSystemID.ClassOKID import *
from ClassesSystemID.ClassERA import ERAFromInitialConditionResponse, ERA
from SystemIDAlgorithms.IdentificationInitialCondition import identificationInitialCondition
from SystemIDAlgorithms.GetMarkovParameters import getMarkovParameters
from SystemIDAlgorithms.GetInitialConditionResponseMarkovParameters import getInitialConditionResponseMarkovParameters
from Plotting.PlotEigenValues import plotEigenValues
from Plotting.PlotSignals import plotSignals
from Plotting.PlotSingularValues import plotSingularValues
from Plotting.PlotMarkovParameters2 import plotMarkovParameters2



## Initial Condition type, Input Signal parameters and Noise
initialCondition = 'Random'
inputSignalName = 'White Noise'
frequency = 5
total_time = 50
noise = True
snr = 1e8



## Define the dynamics
dt = 0.1
mass = 1800
moment_inertia = 200
spring_constant1 = 1500
spring_constant2 = 3000
damping_coefficient1 = 350
damping_coefficient2 = 400
distance1 = 1.4
distance2 = 1.7
force_coefficient1 = 1
force_coefficient2 = 1
measurements1 = ['position', 'velocity', 'acceleration']
measurements2 = ['position', 'velocity', 'acceleration']
Dynamics = AutomobileSystemDynamics(dt, mass, moment_inertia, spring_constant1, spring_constant2, damping_coefficient1, damping_coefficient2, distance1, distance2, force_coefficient1, force_coefficient2, measurements1, measurements2)



## Initial Condition of the Linear System
if initialCondition == 'Zero':
    initial_condition = np.zeros(Dynamics.state_dimension)
if initialCondition == 'Random':
    initial_condition = np.random.randn(Dynamics.state_dimension)
if initialCondition == 'Custom':
    initial_condition = np.array([10, 10, 10, -20])
initial_states = [(initial_condition, 0)]



## Define the corresponding Linear System
if inputSignalName == 'Zero':
    number_experiments = 1
    systems = []
    initial_states = []
    for i in range(number_experiments):
        init_state = [(1*np.random.randn(Dynamics.state_dimension), 0)]
        initial_states.append(init_state)
        if i == 0:
            Sys = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, init_state, 'Automobile System', Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D)
            systems.append(Sys)
        else:
            systems.append(LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, init_state, 'Automobile System', Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D))
else:
    Sys = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, initial_states, 'Automobile System', Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D)



## Parameters of the Input Signal - From input signal parameters
number_steps = total_time * frequency + 1
if inputSignalName == 'Impulse':
    magnitude_impulse = 1 * np.ones(Dynamics.input_dimension)
if inputSignalName == 'White Noise':
    mean = np.zeros(Dynamics.input_dimension)
    standard_deviation = 1 * np.eye(Dynamics.input_dimension)
if inputSignalName == 'Triangle':
    magnitude_peak = np.random.randn(Dynamics.input_dimension)
if inputSignalName == 'Sinusoidal':
    magnitude_sinusoid = np.random.randn(Dynamics.input_dimension)
    frequency_sinusoid = 1 * np.random.randn(Dynamics.input_dimension)
if inputSignalName == 'Combination':
    maximum_ramp = 1 * np.ones(Dynamics.input_dimension)
    exponential_decay_rate = -0.1 * np.ones(Dynamics.input_dimension)
    standard_deviation = 1 * np.eye(Dynamics.input_dimension)



## Define the Input Signal
if inputSignalName == 'Impulse':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName, magnitude_impulse=magnitude_impulse)
if inputSignalName == 'White Noise':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName, mean=mean, standard_deviation=standard_deviation)
if inputSignalName == 'Triangle':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName, magnitude_peak=magnitude_peak)
if inputSignalName == 'Sinusoidal':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName, magnitude_sinusoid=magnitude_sinusoid, frequency_sinusoid=frequency_sinusoid)
if inputSignalName == 'Combination':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName, maximum_ramp=maximum_ramp, standard_deviation=standard_deviation, exponential_decay_rate=exponential_decay_rate, )
if inputSignalName == 'Zero':
    S1 = Signal(total_time, frequency, Dynamics.input_dimension, inputSignalName)
    input_signals = []
    for i in range(number_experiments):
        input_signals.append(S1)



## Define the Output Signal
if inputSignalName == 'Zero':
    Exp = Experiments(systems, input_signals)
    S2_ini = Exp.output_signals[0]
else:
    S2_ini = OutputSignal(S1, Sys, 'Output Signal')



## Add some noise
if noise:
    if inputSignalName == 'Zero':
        for i in range(number_experiments):
            output_signal = Exp.output_signals[i]
            mean_noise = np.zeros(Dynamics.output_dimension)
            standard_deviation_noise = np.eye(Dynamics.output_dimension) * np.mean(output_signal.data ** 2) / snr
            print(standard_deviation_noise)
            SNoise = Signal(total_time, frequency, Dynamics.output_dimension, 'Noise', mean=mean_noise, standard_deviation=standard_deviation_noise)
            Exp.output_signals[i] = add2Signals(output_signal, SNoise)
        S2 = Exp.output_signals[0]
    else:
        mean_noise = np.zeros(Dynamics.output_dimension)
        standard_deviation_noise = np.eye(Dynamics.output_dimension) * np.mean(S2_ini.data**2) / snr
        SNoise = Signal(total_time, frequency, Dynamics.output_dimension, 'Noise', mean=mean_noise, standard_deviation=standard_deviation_noise)
        S2 = add2Signals(S2_ini, SNoise)
else:
    if inputSignalName == 'Zero':
        S2 = Exp.output_signals[0]
    else:
        S2 = S2_ini



## Calculate Markov Parameters and Identified system
if inputSignalName == 'Zero':
    markov_parameters_true = getInitialConditionResponseMarkovParameters(Dynamics.A, Dynamics.C, number_steps)
    ERA1 = ERAFromInitialConditionResponse(Exp.output_signals, Dynamics.state_dimension, Dynamics.input_dimension)
    markov_parameters = getInitialConditionResponseMarkovParameters(ERA1.A, ERA1.C, number_steps)
elif inputSignalName == 'Impulse':
    markov_parameters_true = getMarkovParameters(Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D, number_steps)
    markov_parameters = OKID(S1, S2).markov_parameters
    ERA1 = ERA(markov_parameters, Dynamics.state_dimension)
else:
    markov_parameters_true = getMarkovParameters(Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D, number_steps)
    if initialCondition == 'Zero':
        OKID1 = OKIDObserver(S1, S2)
        markov_parameters = OKID1.markov_parameters
        ERA1 = ERA(markov_parameters, Dynamics.state_dimension)
    else:
        OKID1 = OKIDObserverWithInitialCondition(S1, S2)
        markov_parameters = OKID1.markov_parameters
        ERA1 = ERA(markov_parameters, Dynamics.state_dimension)
        x0 = identificationInitialCondition(S1, S2, ERA1.A, ERA1.B, ERA1.C, ERA1.D, 0)



## Define Identified System
if inputSignalName == 'Zero':
    SysID = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, [(ERA1.x0[:, 0], 0)], 'Identified System', ERA1.A, ERA1.B, ERA1.C, ERA1.D)
elif inputSignalName == 'Impulse':
    SysID = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, initial_states, 'Identified System', ERA1.A, ERA1.B, ERA1.C, ERA1.D)
else:
    if initialCondition == 'Zero':
        SysID = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, initial_states, 'Identified System', ERA1.A, ERA1.B, ERA1.C, ERA1.D)
    else:
        SysID = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, [(x0[:, 0], 0)], 'Identified System', ERA1.A, ERA1.B, ERA1.C, ERA1.D)



## Define the Identified Output Signal
if inputSignalName == 'Zero':
    S2ID = OutputSignal(input_signals[0], SysID, 'Identified Output Signal')
else:
    S2ID = OutputSignal(S1, SysID, 'Identified Output Signal')



## Plotting
if noise:
    plotSignals([[S1], [S2_ini, S2, S2ID], [subtract2Signals(S2, S2ID)]], 1)
else:
    plotSignals([[S1], [S2, S2ID], [subtract2Signals(S2, S2ID)]], 1)
plotEigenValues([Sys, SysID], 2)
plotSingularValues([ERA1], ['IdentifiedSystem'], 3)
plotMarkovParameters2(markov_parameters, markov_parameters_true, 'OKID', 'True', 4)


