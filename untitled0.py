# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:21:35 2020

@author: sihan
"""
#%%
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import os
import math
import pandas as pd

#%%
coordinate = np.loadtxt('10.txt')
velocity_all = np.zeros(coordinate.shape)
force_all = np.zeros(coordinate.shape)
energy_all = np.zeros((coordinate.shape[0],1))
#%%
# Set some basic property 
total_time = 1000
eps = 1
rm = 1
time_step = 0.5
mass = 1

#%%
for one_step in np.arange(0, total_time, time_step):
    # calculate the distance and energy
    for one_row_index in range(coordinate.shape[0]):
        one_row = coordinate[one_row_index, :]
        coordinate_diff_matrix = coordinate - one_row
    #    if one_row_index == 1:
    #        print(distance_matrix)
        coordinate_diff_matrix_square = coordinate_diff_matrix*coordinate_diff_matrix
    #    if one_row_index == 0:
    #        print(coordinate_diff_matrix_square)
        coordinate_diff_matrix_square = coordinate_diff_matrix_square.sum(axis=1).reshape(coordinate_diff_matrix_square.sum(axis=1).shape[0],1)
        distance = np.sqrt(coordinate_diff_matrix_square)
    #    if one_row_index == 0:
    #       print(distance)
        energy = 4 * eps * ((1/(distance+0.0000000000001))**12-(1/(distance+0.000001))**6)
        energy[np.argmax(np.abs(energy)),:]=0
    #    if one_row_index == 0:
    #        print(energy)
        force = 4 * eps * (-12*(1/(distance+0.0000000000001))**(13)+6*(1/(distance+0.000001))**(7))
        force[np.argmax(np.abs(force)),:]=0
        
        # cos matrix
        
        cos_matrix = coordinate_diff_matrix / (distance+0.0000000000001)
        force = force * cos_matrix
        force_all += force
        energy_all += energy
    
    # velocity verlet
    velocity_all = velocity_all + force/mass*(time_step/2)
    # new coordinate
    coordinate = coordinate + velocity_all
    print()
    print(coordinate)
    print()
    
#    ########new force #############
#    for one_row_index in range(coordinate.shape[0]):
#        one_row = coordinate[one_row_index, :]
#        coordinate_diff_matrix = coordinate - one_row
#    #    if one_row_index == 1:
#    #        print(distance_matrix)
#        coordinate_diff_matrix_square = coordinate_diff_matrix*coordinate_diff_matrix
#    #    if one_row_index == 0:
#    #        print(coordinate_diff_matrix_square)
#        coordinate_diff_matrix_square = coordinate_diff_matrix_square.sum(axis=1).reshape(coordinate_diff_matrix_square.sum(axis=1).shape[0],1)
#        distance = np.sqrt(coordinate_diff_matrix_square)
#    #    if one_row_index == 0:
#    #       print(distance)
#        energy = 4 * eps * ((1/(distance+0.0000000000001))**12-(1/(distance+0.000001))**6)
#        energy[np.argmax(np.abs(energy)),:]=0
#    #    if one_row_index == 0:
#    #        print(energy)
#        force = 4 * eps * (-12*(1/(distance+0.0000000000001))**(13)+6*(1/(distance+0.000001))**(7))
#        force[np.argmax(np.abs(force)),:]=0
#    ###############################
#    print(111)
#    velocity_all = velocity_all + force_all/mass * (time_step/2)
#    
#    energy_all = energy_all.sum()/2
#    print(energy_all)