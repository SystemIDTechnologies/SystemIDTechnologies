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



from ClassesSystems.ClassMassSpringDamperDynamics import MassSpringDamperDynamics
from ClassesSystems.ClassTwoMassSpringDamperDynamics import TwoMassSpringDamperDynamics
from ClassesSystems.ClassThreeMassSpringDamperDynamics import *
from ClassesSystems.ClassAutomobileSystemDynamics import AutomobileSystemDynamics
from ClassesGeneral.ClassSystem import LinearSystem
from ClassesGeneral.ClassSignal import Signal, OutputSignal, subtract2Signals
from ClassesGeneral.ClassExperiments import Experiments
from ClassesSystemID.ClassOKID import *
from ClassesSystemID.ClassERA import ERA, ERAFromInitialConditionResponse
from SystemIDAlgorithms.IdentificationInitialCondition import identificationInitialCondition
from SystemIDAlgorithms.GetMarkovParameters import getMarkovParameters
from Plotting.PlotEigenValues import plotEigenValues
from Plotting.PlotSignals import plotSignals
from Plotting.PlotSingularValues import plotSingularValues
from Plotting.PlotMarkovParameters2 import plotMarkovParameters2


## Parameters of the Dynamics - From system parameters
dt = 0.1
mass1 = 1000
mass2 = 1800
mass3 = 2500
moment_inertia = 206
spring_constant1 = 3600
spring_constant2 = 3800
spring_constant3 = 2200
damping_coefficient1 = 60
damping_coefficient2 = 40
damping_coefficient3 = 50
distance1 = 1.4
distance2 = 1.7
force_coefficient1 = 1
force_coefficient2 = 2
force_coefficient3 = 3
measurements1 = ['position', 'velocity', 'acceleration']
measurements2 = ['position', 'velocity', 'acceleration']
measurements3 = ['position', 'velocity', 'acceleration']


## Get an instance of the dynamics
#Dynamics = MassSpringDamperDynamics(dt, mass1, spring_constant1, damping_coefficient1, force_coefficient1, measurements1)
Dynamics = TwoMassSpringDamperDynamics(dt, mass1, mass2, spring_constant1, spring_constant2, damping_coefficient1, damping_coefficient2, force_coefficient1, force_coefficient2, measurements1, measurements2)
#Dynamics = ThreeMassSpringDamperDynamics(dt, mass1, mass2, mass3, spring_constant1, spring_constant2, spring_constant3, damping_coefficient1, damping_coefficient2, damping_coefficient3, force_coefficient1, force_coefficient2, force_coefficient3, measurements1, measurements2, measurements3)
#Dynamics = AutomobileSystemDynamics(dt, mass1, moment_inertia, spring_constant1, spring_constant2, damping_coefficient1, damping_coefficient2, distance1, distance2, force_coefficient1, measurements1, measurements2)


## Parameters of the Linear System
frequency = 2     # From input signal frequency
#initial_condition = np.zeros(Dynamics.state_dimension)
initial_condition = 1*np.random.randn(Dynamics.state_dimension)      # Initial condition of the system - 0 here
initial_states = [(initial_condition, 0)]
name = 'Three Mass Spring Damper System'


## Define the corresponding Linear System
# number_experiments = 5
# systems = []
# for i in range(number_experiments):
#     initial_states = [(1*np.random.randn(Dynamics.state_dimension), 0)]
#     systems.append(LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, initial_states, name, Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D))
Sys = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, initial_states, name, Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D)


## Parameters of the Input Signal - From input signal parameters
total_time = 2000
name = 'Input Signal - White noise'
mean = np.zeros(Dynamics.input_dimension)
standard_deviation = 5 * np.eye(Dynamics.input_dimension)
magnitude_impulse = np.array([10])


## Define the Input Signal
S1 = Signal(total_time, frequency, Dynamics.input_dimension, name, mean=mean, standard_deviation=standard_deviation)
#S1 = Signal(total_time, frequency, Dynamics.input_dimension, name, magnitude_impulse=magnitude_impulse)
# input_signals = []
# for i in range(number_experiments):
#     input_signals.append(Signal(total_time, frequency, Dynamics.input_dimension, name))


## Define the Output Signal
S2 = OutputSignal(S1, Sys, 'Output Signal')
#Exp = Experiments(systems, input_signals)


## Calculate Markov Parameters
markov_parameters = OKIDObserverWithInitialCondition(S1, S2).markov_parameters
markov_parameters_true = getMarkovParameters(Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D, S1.number_steps)
ERA1 = ERA(markov_parameters, Dynamics.state_dimension)
#ERA1 = ERAFromInitialConditionResponse(Exp.output_signals, Dynamics.state_dimension)
x0 = identificationInitialCondition(S1, S2, ERA1.A, ERA1.B, ERA1.C, ERA1.D, 0)


## Define Identified System
SysID = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, [(x0[:, 0], 0)], 'Identified System', ERA1.A, Dynamics.B, ERA1.C, Dynamics.D)


## Define the Identified Output Signal
S2ID = OutputSignal(S1, SysID, 'Identified Output Signal')


## Plotting
plotSignals([[S1], [S2, S2ID], [subtract2Signals(S2, S2ID)]], 1)
plotEigenValues([Sys, SysID], 2)
plotSingularValues([ERA1], ['IdentifiedSystem'], 3)
plotMarkovParameters2(markov_parameters, markov_parameters_true, 'OKID', 'True', 4)



# X0 = np.concatenate((systems[0].initial_states[0][0][:, None],
#                      systems[1].initial_states[0][0][:, None],
#                      systems[2].initial_states[0][0][:, None],
#                      systems[3].initial_states[0][0][:, None],
#                      systems[4].initial_states[0][0][:, None]), axis=1)
# T = np.matmul(X0, LA.pinv(ERA1.x0_id))
# AID = np.matmul(T, np.matmul(ERA1.A(0), inv(T)))