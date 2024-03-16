import numpy as np
import cupy as cp
import torch
if torch.cuda.is_available():
    from cupy.fft import rfftn, irfftn
else:
    from numpy.fft import rfftn, irfftn
    print("GPU acceleration is not available. Change all cp's into np's.")
import math
pi = math.pi; exp = np.exp; log = np.log

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

#pesudo-spectral.py
def zero_cat(X):
    pad = cp.zeros((My,Mx//2+1), dtype = np.complex128)
    X = cp.concatenate((X[:,:],pad[:Ny,-Nx//4:]), axis = 1) # (Mx//2 + 1) - (Nx//2 + 1)
    X = cp.concatenate((X[:Ny//2,:],pad[Ny//2:-Ny//2,:],X[Ny//2:,:]), axis = 0)
    return X

def zero_padding(X):
    X[:,-Nx//4:] = 0; X[Ny//2:-Ny//2,:] = 0
    return X

def removing_interpolation(X):
    X = cp.concatenate((X[:Ny//2,:-Nx//4],X[-Ny//2:,:-Nx//4]), axis = 0)
    return X

def nonlinear_wave(wk): # Pesudo-spectral method with 3/2 zero-padding
    w_xk_pad = zero_cat(1J*KX*wk); w_x_pad = irfftn(w_xk_pad, axes = (0,1), norm = 'forward').real
    w_yk_pad = zero_cat(1J*KY*wk); w_y_pad = irfftn(w_yk_pad, axes = (0,1), norm = 'forward').real

    psik = wk/k_sq; uk,vk = 1J*KY*psik,-1J*KX*psik
    uk_pad = zero_cat(uk); u_pad = irfftn(uk_pad, axes = (0,1), norm = 'forward').real
    vk_pad = zero_cat(vk); v_pad = irfftn(vk_pad, axes = (0,1), norm = 'forward').real
    
    convective_term = u_pad*w_x_pad + v_pad*w_y_pad
    convective_term_wave = rfftn(convective_term, axes = (0,1), norm = 'forward')
    return removing_interpolation(convective_term_wave), uk, vk