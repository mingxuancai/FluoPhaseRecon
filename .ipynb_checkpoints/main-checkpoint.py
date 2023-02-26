import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

# ---------------------------#
# Specify the path of the repo
# ---------------------------#
from opticaltomography.forward import PhaseObject3D, Multislice

# Units in microns
wavelength = 0.6 # fluorescence wavelength
# objective immersion media
n_measure = 1.33
# background refractive index, PDMS
n_b = 1.33
mag = 20
pixel_size = 6.5e-6
na = 0.6
dx = pixel_size / mag
dy = dx
dz = 20 * pixel_size/mag

# import known input field from matlab file
# fx_illu_list: x position ex. 0.0041
# fy_illu_list: y position ex. -0.0146
# fz_illu_list: z layer    ex. 3
na_list = sio.loadmat("na_list_3D_rand_10.mat")
fx_illu_list = na_list["na_list_3D_rand"][:, 0] * 0.9    #paraxial condition
fy_illu_list = na_list["na_list_3D_rand"][:, 1] * 0.9
fz_illu_list = na_list["na_list_3D_rand"][:, 2] * 0      #z layer at the z=0

# initial object
phantom = np.zeros((200, 200, 25), dtype="complex64")

# ---------------------------#
# Setup solver objects
# ---------------------------#
solver_params = dict(wavelength = wavelength, na = na, \
                     RI_measure = n_measure, sigma = 2 * np.pi * dz / wavelength,\
                     fx_illu_list = fx_illu_list, fy_illu_list = fy_illu_list, fz_illu_list = fz_illu_list, \
                     voxel_size=(dy,dx,dz), pad = False, pad_size=(50,50))
phase_obj_3d = PhaseObject3D(shape=phantom.shape, voxel_size=(dy,dx,dz), RI=n_b)
solver_obj = Multislice(phase_obj_3d=phase_obj_3d, **solver_params)

# ---------------------------#
# Fill in phantom
# ---------------------------#
# import the ground truth of the 3D object to generate forward measurements
phantom_coordinates = sio.loadmat("phantomPoisson_nooverlap.mat")
phantom_temp = phantom_coordinates["phantomPoisson"]
solver_obj._x[50:150, 50:150, :] = phantom_temp.astype("complex64") * 0.1