import numpy as np
import cupy as cp
import torch
if torch.cuda.is_available():
    from cupy.fft import rfftn, irfftn
else:
    from numpy.fft import rfftn, irfftn
    print("GPU acceleration is not available. Change all cp's into np's.")
import h5py
import sys
import os
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%m/%d/%Y %H:%M:%S")
import math
pi = math.pi; exp = np.exp; log = np.log

from pesudo_spectral import nonlinear_wave

Data_Dir = 'D:/Research/00_Data/2D_DHIT/resolution 128'; os.makedirs(Data_Dir, exist_ok = True)
Train_Dir = Data_Dir + '/Training data'; os.makedirs(Train_Dir, exist_ok = True)
Val_Dir = Data_Dir + '/Validation data'; os.makedirs(Val_Dir, exist_ok = True)
Test_Dir = Data_Dir + '/Test data'; os.makedirs(Test_Dir, exist_ok = True)

# Parameters
a = 0; b = 2*pi; L = b - a # Box size
N = 128; Nx,Ny = N,N  # Grid resolution
Mx,My = 3*Nx//2,3*Ny//2
dt = 0.0025; T = 3*10**1; Nt = int(T/dt); K = 20  # Time step

# Grid and wavenumbers
x = cp.linspace(a, b, Nx, endpoint = False, dtype = np.float64)
y = cp.linspace(a, b, Ny, endpoint = False, dtype = np.float64)
Y,X = cp.meshgrid(y, x, indexing='ij')

kx = cp.fft.rfftfreq(Nx, 1/Nx)
ky = cp.fft.fftfreq(Ny, 1/Ny)
KY,KX = cp.meshgrid(ky, kx, indexing='ij')

k_sq = KX**2 + KY**2
k_round = cp.round(cp.sqrt(k_sq)).astype(np.int16); k_max = cp.max(k_round)
k_index, k_count = cp.unique(k_round, return_counts = True)
k_sq[0, 0] = 1

# Physical parameters
nu = 1.0e-3  # Kinematic viscosity (adjust as needed)
# rho =



num_train,num_val,num_test = 500,100,50
logging.info(f"{num_train} number of simulations using C-N for viscous term/ 2nd A-B for nonlinear term")
for num in range(num_train):
    w0 = cp.random.normal(loc = 0.0, scale = 1.0, size = (Ny,Nx))
    wk = rfftn(w0, axes = (0,1), norm = 'forward')
    wk = wk/(cp.abs(wk) + 10**-5) # Make amplitudes of all modes to one. 10**-5 prevents dividing by zero.
    wk = ((cp.abs(k_sq - 10.0) + 1)**-2.0)*wk; wk[0,0] = 0
    w0 = irfftn(wk, axes = (0,1), norm = 'forward').real; w = np.expand_dims(cp.asnumpy(w0), axis = (0,1))
    
    # Time-stepping loop
    for q in range(Nt):
        Jk, uk, vk = nonlinear_wave(wk)
        if q == 0: Jkp = Jk[:]
        wk = ((1 - 0.5*nu*k_sq*dt)*wk - dt*(1.5*Jk - 0.5*Jkp))/(1 + 0.5*nu*k_sq*dt); wk[0,0] = 0
        Jkp[:] = Jk[:]

        if (q + 1) % K == 0:
            wn = irfftn(wk, axes = (0,1), norm = 'forward').real; wn = np.expand_dims(cp.asnumpy(wn), axis = (0,1))
            w = np.concatenate((w,wn), axis = 0)
    
    CreateData = os.path.join(Train_Dir, f"Training{num+1} 2D HIT n={N} T={T} dt={dt:.4f} K={K} data.h5")
    fc = h5py.File(CreateData, 'w')
    fc.create_dataset('w', data = w) # Keyword of data
    fc.close()
    
    logging.info(f"Simulation {num+1} is done.")
