# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import os
import math

# Define the particle3D class
print(os.getcwd())
#%%
class particle3D:
    def __init__(self, coordinates=[0, 0, 0], velocity=[0, 0, 0], mass=1, force=[0, 0, 0]):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.z = coordinates[2]
        self.coordinate = coordinates
        
        self.velo_x = velocity[0]
        self.velo_y = velocity[1]
        self.velo_z = velocity[2]
        self.velocity = velocity
  
        self.mass = mass
    
        self.force_x = force[0]
        self.force_y = force[1]
        self.force_z = force[2]
        self.force = force
    
    def calculat_accelerate(self, external_force):
        return external_force / self.mass
#%%
# Read the data
coordinate = np.loadtxt('10.txt')
particles = []

# Read the data and 
for each_row in range(coordinate.shape[0]):
    particle = particle3D(coordinate[each_row,:])
    particles.append(particle)
#    print(particle.force)
    
#particles = np.array(particles).reshape(1, len(particles))
print(particles)
#%%

# Define some basic parameter
total_time = 1000
time_step = 0.002
eps = 1
rm = 1

# first loop for time step
for one_step in np.arange(0, total_time, time_step):
    # calculate the force and potential energy for all atoms
    count = 0
    
    for one_particle in particles:
        temporary_list = particles.copy()
        temporary_list.pop(count)
        other_particles = temporary_list
        count += 1
        
        # calculate the LJ energy between each the particles and other particles
        
        # calculate the distance
        one_particle_coordinate = np.array(one_particle.coordinate).reshape(1, len(one_particle.coordinate))
        
        energy = np.empty((1,0))
        for another_one_particle in other_particles:
#            energy = 4*eps*((rm/r)**12 - (rm/r)**6)
            # calculate the distance
            distance = math.sqrt((one_particle.x-another_one_particle.x)**2 + (one_particle.y - another_one_particle.y)**2 + (one_particle.z-another_one_particle.z)**2)
            energy = 4*eps*((1/distance)**12 - (1/distance)**6)
            # the force create by this atom
            Force = 4*eps*(-12*distance**(-13)+6*distance**(-7))
            Force_x = Force * (another_one_particle.x-one_particle.x) / distance
            Force_y = Force * (another_one_particle.y-one_particle.y) / distance
            Force_z = Force * (another_one_particle.z-one_particle.z) / distance
            
            # uodate the one_particle force
            for componet in one_particle.force:
                print('doing')
                one_particle.force[0] += Force_x
                one_particle.force[1] += Force_y
                one_particle.force[2] += Force_z
                
                
print(particles)
for each in particles:
    print(each)
#%%
        
        

