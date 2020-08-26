"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np
from numpy import linalg as LA
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt



from ClassesSystems.ClassMassSpringDamperDynamics import MassSpringDamperDynamics
from ClassesSystems.ClassTwoMassSpringDamperDynamics import TwoMassSpringDamperDynamics
from ClassesSystems.ClassThreeMassSpringDamperDynamics import *
from ClassesSystems.ClassAutomobileSystemDynamics import AutomobileSystemDynamics
from ClassesSystems.ClassCustomizedDiscreteTimeInvariantDynamics import CustomizedDiscreteTimeInvariantDynamics
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


## Get the system and initial condition type
systemName = 'Three Mass Spring Damper System'
initialCondition = 'Random'
inputSignalName = 'White Noise'
noise = True



## Define the dynamics
if systemName == 'Mass Spring Damper System':
    dt = 0.1
    mass = 1
    spring_constant = 1
    damping_coefficient = 0.01
    force_coefficient = 1
    measurements = ['position', 'velocity', 'acceleration']
    Dynamics = MassSpringDamperDynamics(dt, mass, spring_constant, damping_coefficient, force_coefficient, measurements)

if systemName == 'Two Mass Spring Damper System':
    dt = 0.1
    mass1 = 1
    mass2 = 1.8
    spring_constant1 = 1
    spring_constant2 = 3
    damping_coefficient1 = 0.3
    damping_coefficient2 = 0.4
    force_coefficient1 = 1
    force_coefficient2 = 2
    measurements1 = ['position', 'velocity', 'acceleration']
    measurements2 = ['position', 'velocity', 'acceleration']
    Dynamics = TwoMassSpringDamperDynamics(dt, mass1, mass2, spring_constant1, spring_constant2, damping_coefficient1, damping_coefficient2, force_coefficient1, force_coefficient2, measurements1, measurements2)

if systemName == 'Three Mass Spring Damper System':
    dt = 0.1
    mass1 = 1
    mass2 = 1.8
    mass3 = 1.4
    spring_constant1 = 1
    spring_constant2 = 3
    spring_constant3 = 2.2
    damping_coefficient1 = 0.3
    damping_coefficient2 = 0.4
    damping_coefficient3 = 0.05
    force_coefficient1 = 1
    force_coefficient2 = 2
    force_coefficient3 = 1.5
    measurements1 = ['position', 'velocity', 'acceleration']
    measurements2 = ['position', 'velocity', 'acceleration']
    measurements3 = ['position', 'velocity', 'acceleration']
    Dynamics = ThreeMassSpringDamperDynamics(dt, mass1, mass2, mass3, spring_constant1, spring_constant2, spring_constant3, damping_coefficient1, damping_coefficient2, damping_coefficient3, force_coefficient1, force_coefficient2, force_coefficient3, measurements1, measurements2, measurements3)

if systemName == 'Automobile system':
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

if systemName == 'Customized':
    Ad = np.array([[0.5, 0.5], [0, 1]])
    Bd = np.array([[0], [1]])
    Cd = np.eye(2)
    Dd = np.array([[0.1], [0.1]])
    Dynamics = CustomizedDiscreteTimeInvariantDynamics(Ad, Bd, Cd, Dd)


## Initial Condition of the Linear System
frequency = 5    # From input signal frequency
if initialCondition == 'Zero':
    initial_condition = np.zeros(Dynamics.state_dimension)
if initialCondition == 'Random':
    initial_condition = np.random.randn(Dynamics.state_dimension)
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
            Sys = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, init_state, systemName, Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D)
            systems.append(Sys)
        else:
            systems.append(LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, init_state, systemName, Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D))
else:
    Sys = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, initial_states, systemName, Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D)



## Parameters of the Input Signal - From input signal parameters
total_time = 50
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
    frequency_sinusoid = 100000 * np.random.randn(Dynamics.input_dimension)
if inputSignalName == 'Combination':
    maximum_ramp = 10 * np.ones(Dynamics.input_dimension)
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
else:
    S2_ini = OutputSignal(S1, Sys, 'Output Signal')



## Add some noise
if noise:
    snr = 1e4
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


#
# # Plotting
# plotSignals([[S1], [S2_ini, S2, S2ID], [subtract2Signals(S2, S2ID)]], 1)
# plotEigenValues([Sys, SysID], 2)
# plotSingularValues([ERA1], ['IdentifiedSystem'], 3)
# plotMarkovParameters2(markov_parameters, markov_parameters_true, 'OKID', 'True', 4)


# Nuclear norm
#n=28
alpha = 0.000000000288
n_iter = 1
r, c = ERA1.H0.shape
print(ERA1.H0.shape)
legend = ['initial']

orders_list = np.zeros([min(r, c), n_iter])

significant = []
unsignificant = []

(R, sigma, St) = LA.svd(ERA1.H0, full_matrices=True)

change = False

# plt.figure(401, figsize=[10, 8])
# plt.semilogy(sigma, 'o')

for i in range(n_iter):
    print('i = ', i)
    H0e = cp.Variable(shape=ERA1.H0.shape)
    objective = cp.Minimize(cp.norm(H0e, "nuc"))
    constraints = [cp.norm(ERA1.H0 - H0e, "fro") / np.sqrt(r*c) <= alpha * (np.mean(np.diag(LA.inv(np.matmul(OKID1.U.T, OKID1.U)))) * np.mean(np.diag(standard_deviation_noise)))]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    (Re, sigmae, Ste) = LA.svd(H0e.value, full_matrices=True)
    orders = np.floor(np.log10([sigmae]))[0]
    orders_list[:, i] = sigmae
    # index_sep = 29
    # for j in range(1, len(orders)):
    #     if orders[j] <= orders[j-1] - 5:
    #         index_sep = j
    #         change = True
    #         if j == len(orders)-1 and change:
    #             index_sep = j
    # significant.append(index_sep)
    # unsignificant.append(len(orders)-index_sep)
    plt.semilogy(sigmae, '*')
    legend.append('alpha = ' + str(alpha))
    alpha = alpha + 0.5

plt.legend(legend, loc='upper right')
# plt.show()


# plt.figure(501)
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[0, :])
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[1, :])
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[2, :])
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[3, :])
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[4, :])
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[5, :])
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[6, :])
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[7, :])
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[8, :])
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[9, :])
# plt.semilogy(np.linspace(1, n_iter, n_iter), orders_list[10, :])
# plt.show()


#print('Frobenius norm:', LA.norm(H0 - H0e.value)/(r*c))



#plt.semilogx([10, 20, 50, 100, 1000, 10000, 100000, 1000000], [0.11, 0.065, 0.09, 0.068, 0.068, 0.065, 0.065, 0.067])
#plt.show()

## Plotting PhD
plt.figure(1, figsize=(5, 4))
plt.semilogy(sigma, 'o', color=(0 / 255, 162 / 255, 255 / 255))
plt.semilogy(sigmae, '*', color=(203 / 255, 41 / 255, 123 / 255))
legend = ['Original SVD', 'Optimized SVD']
plt.legend(legend, loc='upper right')
plt.ylabel('Value of singular values')
plt.xlabel('# of singular values')
plt.show()











# plt.subplot(3, 2, 1)
# plt.plot(time[0:120], S1_nom.data[1, 0:120], color=(0 / 255, 0 / 255, 255 / 255))
# plt.plot(time[0:120], S1_test.data[1, 0:120], color=(0 / 255, 162 / 255, 255 / 255))
# plt.ylabel('u, du')
# plt.xlabel('Time [s]')
# plt.legend(['Nominal input', 'Input deviation'], loc='upper right')
# plt.subplot(3, 2, 3)
# plt.plot(time[0:120], S2_test.data[0, 0:120], color=(0 / 255, 0 / 255, 255 / 255))
# plt.plot(time[0:120], S2ID_test.data[0, 0:120], color=(127 / 255, 0 / 255, 255 / 255))
# plt.plot(time[0:120], S2ID_True.data[0, 0:120], color=(203 / 255, 41 / 255, 123 / 255))
# plt.ylabel('y1')
# plt.xlabel('Time [s]')
# plt.legend(['True trajectory', 'Identified trajectory', 'True linearized trajectory'], loc='upper right')
# plt.subplot(3, 2, 4)
# plt.plot(time[0:120], S2_test.data[1, 0:120], color=(0 / 255, 0 / 255, 255 / 255))
# plt.plot(time[0:120], S2ID_test.data[1, 0:120], color=(127 / 255, 0 / 255, 255 / 255))
# plt.plot(time[0:120], S2ID_True.data[1, 0:120], color=(203 / 255, 41 / 255, 123 / 255))
# plt.ylabel('y2')
# plt.xlabel('Time [s]')
# plt.legend(['True trajectory', 'Identified trajectory', 'True linearized trajectory'], loc='upper right')
# plt.subplot(3, 2, 5)
# plt.plot(time[0:120], np.abs(subtract2Signals(S2_test, S2ID_test).data[0, 0:120]), color=(127 / 255, 0 / 255, 255 / 255))
# plt.plot(time[0:120], np.abs(subtract2Signals(S2_test, S2ID_True).data[0, 0:120]), color=(203 / 255, 41 / 255, 123 / 255))
# plt.ylabel('Absolute Error y1')
# plt.xlabel('Time [s]')
# plt.legend(['Error identified trajectory', 'Error true linearized trajectory'], loc='upper right')
# plt.subplot(3, 2, 6)
# plt.plot(time[0:120], np.abs(subtract2Signals(S2_test, S2ID_test).data[1, 0:120]), color=(127 / 255, 0 / 255, 255 / 255))
# plt.plot(time[0:120], np.abs(subtract2Signals(S2_test, S2ID_True).data[1, 0:120]), color=(203 / 255, 41 / 255, 123 / 255))
# plt.ylabel('Absolute Error y2')
# plt.xlabel('Time [s]')
# plt.show()


