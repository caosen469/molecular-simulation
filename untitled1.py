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
coordinate = np.loadtxt('11.txt')
velocity_all = np.zeros(coordinate.shape)
force_all = np.zeros(coordinate.shape)
energy_all = np.zeros((coordinate.shape[0],1))

energy_record = np.empty((1,0))
#%%
# Set some basic property 
total_time = 10
eps = 1
rm = 1
time_step = 0.001
mass = 1

#for one_step in np.arange(0, total_time, time_step):
    #%%
    # 通过目前的config，根据LJ potential 计算出力和能量
    #%%
    # calculate the distance and energy
    # 遍历所有微粒， 每个循环里面是一个粒子

#更新半步长的速度
velocity_all = velocity_all + force_all/mass*(time_step/2)
 #更新位置
coordinate += velocity_all * time_step

for one_row_index in range(coordinate.shape[0]):
    
    # 计算这个粒子和其他粒子，包括他自己的距离
    one_row = coordinate[one_row_index, :]
#    print(one_row)
    coordinate_diff_matrix = coordinate - one_row
#    print(coordinate_diff_matrix)
#    if one_row_index == 1:
#        print(distance_matrix)
    coordinate_diff_matrix_square = coordinate_diff_matrix*coordinate_diff_matrix
#    print(coordinate_diff_matrix_square)
#    if one_row_index == 0:
#        print(coordinate_diff_matrix_square)
    coordinate_diff_matrix_square = coordinate_diff_matrix_square.sum(axis=1).reshape(coordinate_diff_matrix_square.sum(axis=1).shape[0],1)
    distance = np.sqrt(coordinate_diff_matrix_square)
#        print(distance[0,:])
#    print(distance)
#    if one_row_index == 2:
#       print(distance)
    #%%

    # 计算这个粒子，和其他粒子 包括他自己的 LJ 势能
    energy = 4 * eps * ((1/(distance+0.0000000000001))**12-(1/(distance+0.000001))**6)
    # 把和自己的能量变成0
    energy[np.argmax(np.abs(energy)),:]=0
    
#    if one_row_index == 2:
#        print(energy
              
    energy = energy.sum(axis=0)        
    energy_all[one_row_index] += energy
    
#    print(energy_all)
    #%%

    # 计算这个粒子和其他粒子之间的力，把和自己的力变成0
    force = 48 * ((distance+0.0000000000001)**(-13)-0.5*(distance+0.0000000000001)**(-7))
    force[np.argmax(np.abs(force)),:]=0
#    if one_row_index == 3:
#        print(force)
    
    # cos matrix
    
    # 为了计算里的分量，首先计算矢量的方向余弦
    cos_matrix = coordinate_diff_matrix / (distance+0.0000000000001)
    
    # 将力分解到三个方向
    force = force * cos_matrix
#    if one_row_index == 2:
#        print()
#        print(force)
#        print()
#    # 计算其他分子对这个分子产生的力
    force = force.sum(axis=0)
#    print(force)
#    if one_row_index == 2:
#        print()
#        print(force)
#        print()
    force_all[one_row_index] += force
#    print(force_all)
    
#    if one_row_index == 2:
#        print()
#        print(force_all)W
#        print()
    #%% 计算这个布局的能量
#    energy_all += energy
#    energy_all /= 2
#    print(energy_all)
    # energy of the system at this configuration
total_energy = energy_all.sum()/2
#print(total_energy)
#print(force_all)
energy_record = np.concatenate((energy_record, np.array([[total_energy]])), axis=1) 
#print(distance)
#    print(energy_record)
#    plt.plot(energy_record)
#%%
# 上述代码主要是    更新了所有分子的受力，以及能量
# 下面的代码是：
    #%%
    ############### velocity verlet###########
# velocity verlet

# 半个时间步以后的速度
#print(velocity_all)
#velocity_all = velocity_all + force_all/mass*(time_step/2)
#print(velocity_all)

#print(velocity_all)
# new coordinate
#%%
# 一个时间步后位置
# coordinate = coordinate + velocity_all
#print(coordinate[0,1])
#print(coordinate)
#%%
########new force #############

#计算的新的力和能量
#%% 新的分之间的距离
for one_row_index in range(coordinate.shape[0]):
    
    # 计算这个粒子和其他粒子，包括他自己的距离
    one_row = coordinate[one_row_index, :]
    coordinate_diff_matrix = coordinate - one_row
#    if one_row_index == 1:
#        print(distance_matrix)
    coordinate_diff_matrix_square = coordinate_diff_matrix*coordinate_diff_matrix
#    if one_row_index == 0:
#        print(coordinate_diff_matrix_square)
    coordinate_diff_matrix_square = coordinate_diff_matrix_square.sum(axis=1).reshape(coordinate_diff_matrix_square.sum(axis=1).shape[0],1)
    distance = np.sqrt(coordinate_diff_matrix_square)
#    print(distance)
#    if one_row_index == 2:
#       print(distance)



#%% 计算新的力
    
    force = 48 * ((distance+0.0000000000001)**(-13)-0.5*(distance+0.0000000000001)**(-7))
    force[np.argmax(np.abs(force)),:]=0
    cos_matrix = coordinate_diff_matrix / (distance+0.0000000000001)
    force = force * cos_matrix
    force = force.sum(axis=0)
    force_all[one_row_index] += force
#    print(force_all)
#%%
velocity_all = velocity_all + force_all/mass*(time_step/2)
#    print(force_all[0,:])


    

    