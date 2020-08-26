#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:27:31 2019

@author: djg76
"""


import numpy as np
from numpy import linalg as LA
from scipy.linalg import fractional_matrix_power as matpow


import PlotSingularValues as PlotSingularValues


def ERADC(Markov_list, tau, n):
    
    
    ## Sizes
    alpha = int(np.floor((len(Markov_list)-1)/(2*(1+tau))))
    xi = alpha
    gamma = 1+2*xi*tau
    if Markov_list[0].shape==():
        (m, r) = (1, 1)
    else:
        (m, r) = Markov_list[0].shape
        
        
    ## Building Hankel Matrices
    H = np.zeros([alpha*m, alpha*r, gamma+1])
    for i in range(alpha):
        for j in range(alpha):
            for k in range(gamma+1):
                H[i*m:(i+1)*m, j*r:(j+1)*r, k] = Markov_list[i+j+1+k]
            
            
    ## Building Data Correlation Matrices
    R = np.zeros([alpha*m, alpha*m, gamma+1])
    for i in range(gamma+1):
        R[:, :, i] = np.matmul(H[:, :, i], np.transpose(H[:, :, 0]))
        
        
    (c, scc, cc) = LA.svd(R[:, :, 0], full_matrices=True)
    PlotSingularValues.PlotSingularValues(scc, 'scc', 'r')
    
    
    ## Building Block Correlation Hankel Matrices
    H0 = np.zeros([(xi+1)*alpha*m, (xi+1)*alpha*m])
    H1 = np.zeros([(xi+1)*alpha*m, (xi+1)*alpha*m])
    for i in range(xi+1):
        for j in range(xi+1):
            H0[i*alpha*m:(i+1)*alpha*m, j*alpha*m:(j+1)*alpha*m] = R[:, :, (i+j)*tau]
            H1[i*alpha*m:(i+1)*alpha*m, j*alpha*m:(j+1)*alpha*m] = R[:, :, (i+j)*tau+1]
            
     
#    R0 = R[:, :, 0]
#    h0 = np.zeros([alpha*m, alpha*r])
#    for i in range(alpha):
#        for j in range(i+1):
#            h0[i*m:(i+1)*m, j*r:(j+1)*r] = Markov_list[i-j]
#    T = R0 - np.matmul(h0, np.transpose(h0))
#    
#    (t1, sv, t2) = LA.svd(T, full_matrices=True)
#    PlotSingularValues.PlotSingularValues(sv, 'T', 'b')
        
        
        
    ## SVD H(0)
    (R, sigma, St) = LA.svd(H0, full_matrices=True)
    PlotSingularValues.PlotSingularValues(sigma, 'ERA/DC', 'r')
    Sigma = np.diag(sigma)
    
    
    ## Matrices Rn, Sn, Sigman
    Rn = R[:, 0:n]
    Snt = St[0:n, :]
    Sigman = Sigma[0:n, 0:n]
    
    
    ## Identified matrices
    A_id = np.matmul(matpow(Sigman, -1/2), np.matmul(np.transpose(Rn), np.matmul(H1, np.matmul(np.transpose(Snt), matpow(Sigman, -1/2)))))
    B_temp1 = np.matmul(Rn, matpow(Sigman, 1/2))
    B_temp2 = B_temp1[0:alpha*m, :]
    B_temp3 = np.matmul(LA.pinv(B_temp2), H[:, :, 0])
    B_id = B_temp3[:, 0:r]
    C_temp = np.matmul(Rn, matpow(Sigman, 1/2))
    C_id = C_temp[0:m, :]
    D_id = Markov_list[0]
    
    
    return (A_id, B_id, C_id, D_id)
    