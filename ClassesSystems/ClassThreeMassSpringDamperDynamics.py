"""
Author: Damien GUEHO
Copyright: Copyright (C) 2020 Damien GUEHO
License: Public Domain
Version: 1.0.0
Date: August 2020
Python: 3.7.7
"""


import numpy as np
from numpy.linalg import inv, matrix_power
from scipy.linalg import expm, eig


class ThreeMassSpringDamperDynamics:
    def __init__(self, dt, mass1, mass2, mass3, spring_constant1, spring_constant2, spring_constant3, damping_coefficient1, damping_coefficient2, damping_coefficient3, force_coefficient1, force_coefficient2, force_coefficient3, measurements1, measurements2, measurements3):
        self.state_dimension = 6
        self.input_dimension = 3
        self.output_dimension = len(measurements1) + len(measurements2) + len(measurements3)
        self.dt = dt
        self.frequency = 1 / dt
        self.mass1 = mass1
        self.mass2 = mass2
        self.mass3 = mass3
        self.spring_constant1 = spring_constant1
        self.spring_constant2 = spring_constant2
        self.spring_constant3 = spring_constant3
        self.damping_coefficient1 = damping_coefficient1
        self.damping_coefficient2 = damping_coefficient2
        self.damping_coefficient3 = damping_coefficient3
        self.force_coefficient1 = force_coefficient1
        self.force_coefficient2 = force_coefficient2
        self.force_coefficient3 = force_coefficient3
        self.measurements1 = measurements1
        self.measurements2 = measurements2
        self.measurements3 = measurements3
        self.total_measurements = []
        self.units = []
        self.M = np.zeros([3, 3])
        self.K = np.zeros([3, 3])
        self.Z = np.zeros([3, 3])
        self.M[0, 0] = self.mass1
        self.M[1, 1] = self.mass2
        self.M[2, 2] = self.mass3
        self.K[0, 0] = self.spring_constant1 + self.spring_constant3
        self.K[0, 1] = -self.spring_constant1
        self.K[0, 2] = -self.spring_constant3
        self.K[1, 0] = -self.spring_constant1
        self.K[1, 1] = self.spring_constant1 + self.spring_constant2
        self.K[1, 2] = -self.spring_constant2
        self.K[2, 0] = -self.spring_constant3
        self.K[2, 1] = -self.spring_constant2
        self.K[2, 2] = self.spring_constant2 + self.spring_constant3
        self.Z[0, 0] = self.damping_coefficient1 + self.damping_coefficient3
        self.Z[0, 1] = -self.damping_coefficient1
        self.Z[0, 2] = -self.damping_coefficient3
        self.Z[1, 0] = -self.damping_coefficient1
        self.Z[1, 1] = self.damping_coefficient1 + self.damping_coefficient2
        self.Z[1, 2] = -self.damping_coefficient2
        self.Z[2, 0] = -self.damping_coefficient3
        self.Z[2, 1] = -self.damping_coefficient2
        self.Z[2, 2] = self.damping_coefficient2 + self.damping_coefficient3

        self.Ac = np.zeros([self.state_dimension, self.state_dimension])
        self.Ac[0:3, 3:6] = np.eye(3)
        self.Ac[3:6, 0:3] = np.matmul(-inv(self.M), self.K)
        self.Ac[3:6, 3:6] = np.matmul(-inv(self.M), self.Z)
        self.Ad = expm(self.Ac * self.dt)

        self.B2 = np.zeros([int(self.state_dimension / 2), self.input_dimension])
        self.B2[0, 0] = self.force_coefficient1
        self.B2[1, 1] = self.force_coefficient2
        self.B2[2, 2] = self.force_coefficient3
        self.Bc = np.zeros([self.state_dimension, self.input_dimension])
        self.Bc[3:6, 0:3] = np.matmul(inv(self.M), self.B2)
        self.Bd = np.eye(6) * self.dt
        for i in range(1, 100):
            self.Bd = self.Bd + matrix_power(self.Ac, i)*self.dt**(i+1)/np.math.factorial(i+1)
        self.Bd = np.matmul(self.Bd, self.Bc)

        self.Cd = np.zeros([self.output_dimension, self.state_dimension])
        self.Cp = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        self.Cv = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        self.Ca = np.zeros([self.output_dimension, int(self.state_dimension / 2)])
        i = 0
        if 'position' in self.measurements1:
            self.Cp[i, 0] = 1
            i += 1
            self.total_measurements.append('Position 1')
            self.units.append('m')
        if 'position' in self.measurements2:
            self.Cp[i, 1] = 1
            i += 1
            self.total_measurements.append('Position 2')
            self.units.append('m')
        if 'position' in self.measurements3:
            self.Cp[i, 2] = 1
            i += 1
            self.total_measurements.append('Position 3')
            self.units.append('m')
        if 'velocity' in self.measurements1:
            self.Cv[i, 0] = 1
            i += 1
            self.total_measurements.append('Velocity 1')
            self.units.append('m/s')
        if 'velocity' in self.measurements2:
            self.Cv[i, 1] = 1
            i += 1
            self.total_measurements.append('Velocity 2')
            self.units.append('m/s')
        if 'velocity' in self.measurements3:
            self.Cv[i, 2] = 1
            i += 1
            self.total_measurements.append('Velocity 3')
            self.units.append('m/s')
        if 'acceleration' in self.measurements1:
            self.Ca[i, 0] = 1
            i += 1
            self.total_measurements.append('Acceleration 1')
            self.units.append('m/s^2')
        if 'acceleration' in self.measurements2:
            self.Ca[i, 1] = 1
            i += 1
            self.total_measurements.append('Acceleration 2')
            self.units.append('m/s^2')
        if 'acceleration' in self.measurements3:
            self.Ca[i, 2] = 1
            i += 1
            self.total_measurements.append('Acceleration 3')
            self.units.append('m/s^2')
        self.Cd[:, 0:int(self.state_dimension / 2)] = self.Cp - np.matmul(self.Ca, np.matmul(inv(self.M), self.K))
        self.Cd[:, int(self.state_dimension / 2): self.state_dimension] = self.Cv - np.matmul(self.Ca, np.matmul(inv(self.M), self.Z))

        self.Dd = np.matmul(self.Ca, np.matmul(inv(self.M), self.B2))


    def A(self, tk):
        return self.Ad

    def B(self, tk):
        return self.Bd

    def C(self, tk):
        return self.Cd

    def D(self, tk):
        return self.Dd
