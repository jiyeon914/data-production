import numpy as np
import cupy as cp
import torch
if torch.cuda.is_available():
    from cupy.fft import rfftn, irfftn
else:
    from numpy.fft import rfftn, irfftn
    print("GPU acceleration is not available. Change all cp's into np's.")
    
#pesudo-spectral.py
def zero_cat(X):
    pad = cp.zeros_like((My,Mx//2+1), dtype = np.complex128)
    X = cp.concatenate((X[:,:],pad[:Ny,-Nx//2:]), axis = 1) # (Mx//2 + 1) - (Nx//2 + 1)
    X = cp.concatenate((X[:Ny//2,:],pad[Ny//2:-Ny//2,:],X[Ny//2:,:]), axis = 0)
    return X

def zero_padding(X):
    X[:,-Nx//2:] = 0; X[Ny//2:-Ny//2,:] = 0
    return X

def removing_interpolation(X):
    X = cp.concatenate((X[:Ny//2,:-Nx//2],X[-Ny//2:,:-Nx//2]), axis = 0)
    return X

def nonlinear_wave(wk): # Pesudo-spectral method with 3/2 zero-padding
    w_xk_pad = zero_cat(1J*KX*wk); w_x_pad = irfftn(w_xk_pad, axes = (0,1), norm = 'forward').real
    w_yk_pad = zero_cat(1J*KY*wk); w_y_pad = irfftn(w_yk_pad, axes = (0,1), norm = 'forward').real

    psik = wk/k_sq; uk,vk = 1J*KY*psik,-1J*KX*psik
    uk_pad = zero_cat(uk); u_pad = irfftn(uk_pad, axes = (0,1), norm = 'forward').real
    vk_pad = zero_cat(vk); v_pad = irfftn(vk_pad, axes = (0,1), norm = 'forward').real
    
    convective_term = u_pad*w_x_pad + v_pad*w_y_pad
    convective_term_wave = rttfn(convective_term, axes = (0,1), norm = 'forward')
    return removing_interpolation(convective_term_wave), uk, vk