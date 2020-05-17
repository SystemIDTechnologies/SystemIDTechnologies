"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 0.1
Date: May 2020
Python: 3.7.7
"""


import numpy as np


from ClassesSystems.ClassMassSpringDamperDynamics import MassSpringDamperDynamics
from ClassesSystems.ClassTwoMassSpringDamperDynamics import TwoMassSpringDamperDynamics
from ClassesSystems.ClassThreeMassSpringDamperDynamics import *
from ClassesSystems.ClassAutomobileSystemDynamics import AutomobileSystemDynamics
from ClassesGeneral.ClassSystem import LinearSystem
from ClassesGeneral.ClassSignal import Signal, OutputSignal, subtract2Signals
from ClassesSystemID.ClassOKID import *
from ClassesSystemID.ClassERA import ERA
from SystemIDAlgorithms.IdentificationInitialCondition import identificationInitialCondition
from Plotting.PlotEigenValues import plotEigenValues
from Plotting.PlotSignals import plotSignals
from Plotting.PlotSingularValues import plotSingularValues


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
#Dynamics = TwoMassSpringDamperDynamics(dt, mass1, mass2, spring_constant1, spring_constant2, damping_coefficient1, damping_coefficient2, force_coefficient1, force_coefficient2, measurements1, measurements2)
Dynamics = ThreeMassSpringDamperDynamics(dt, mass1, mass2, mass3, spring_constant1, spring_constant2, spring_constant3, damping_coefficient1, damping_coefficient2, damping_coefficient3, force_coefficient1, force_coefficient2, force_coefficient3, measurements1, measurements2, measurements3)
#Dynamics = AutomobileSystemDynamics(dt, mass1, moment_inertia, spring_constant1, spring_constant2, damping_coefficient1, damping_coefficient2, distance1, distance2, force_coefficient1, measurements1, measurements2)


## Parameters of the Linear System
frequency = 5     # From input signal frequency
#initial_condition = np.zeros(Dynamics.state_dimension)
initial_condition = 100*np.random.randn(Dynamics.state_dimension)      # Initial condition of the system - 0 here
initial_states = [(initial_condition, 0)]
name = 'Three Mass Spring Damper System'


## Define the corresponding Linear System
Sys = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, initial_states, name, Dynamics.A, Dynamics.B, Dynamics.C, Dynamics.D)


## Parameters of the Input Signal - From input signal parameters
total_time = 100
name = 'Input Signal - White noise'
mean = np.array([1])
standard_deviation = 500 * np.eye(Dynamics.input_dimension)
magnitude_impulse = np.array([10])


## Define the Input Signal
S1 = Signal(total_time, frequency, Dynamics.input_dimension, name, mean=mean, standard_deviation=standard_deviation)
#S1 = Signal(total_time, frequency, Dynamics.input_dimension, name, magnitude_impulse=magnitude_impulse)


## Define the Output Signal
S2 = OutputSignal(S1, Sys, 'Output Signal')


## Calculate Markov Parameters
markov_parameters = OKIDObserver(S1, S2).markov_parameters
ERA1 = ERA(markov_parameters, Dynamics.state_dimension)
x0 = identificationInitialCondition(S1, S2, ERA1.A, ERA1.B, ERA1.C, ERA1.D, 0)


## Define Identified System
SysID = LinearSystem(frequency, Dynamics.state_dimension, Dynamics.input_dimension, Dynamics.output_dimension, [(x0[:, 0], 0)], 'Identified System', ERA1.A, ERA1.B, ERA1.C, ERA1.D)


## Define the Identified Output Signal
S2ID = OutputSignal(S1, SysID, 'Identified Output Signal')


## Plotting
plotSignals([[S1], [S2, S2ID], [subtract2Signals(S2, S2ID)]], 1)
plotEigenValues([Sys, SysID], 2)
plotSingularValues([ERA1], ['IdentifiedSystem'], 3)
