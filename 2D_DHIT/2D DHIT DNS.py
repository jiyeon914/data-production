import numpy as np
from numpy.fft import fftn as fftn_cpu
from numpy.fft import ifftn as ifftn_cpu
import cupy as cp
from cupy.fft import fftn, ifftn
import h5py
import sys
import os
import logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%m/%d/%Y %H:%M:%S")
import math
pi = math.pi; exp = np.exp; log = np.log

Data_Dir = '/data/jykim3994/0.Data/3D HIT/resolution 128'
Train_Dir = Data_Dir + '/Training data'
Val_Dir = Data_Dir + '/Validation data'
Test_Dir = Data_Dir + '/Test data'
os.makedirs(Data_Dir, exist_ok = True); os.makedirs(Train_Dir, exist_ok = True); os.makedirs(Val_Dir, exist_ok = True); os.makedirs(Test_Dir, exist_ok = True)

# Parameters
a = 0; b = 2*pi; L = b - a # Box size
N = 128; Nx, Ny, Nz = N, N, N  # Grid resolution
Mx, My, Mz = 3*Nx//2, 3*Ny//2, 3*Nz//2
dt = 0.0025; T = 5*10**1; Nt = int(T/dt); K = 20  # Time step

# Grid and wavenumbers
x = cp.linspace(a, b, Nx, endpoint = False, dtype = np.float64)
y = cp.linspace(a, b, Ny, endpoint = False, dtype = np.float64)
z = cp.linspace(a, b, Nz, endpoint = False, dtype = np.float64)
Z, Y, X = cp.meshgrid(z, y, x, indexing='ij')

kx = cp.fft.fftfreq(Nx, 1/Nx)
ky = cp.fft.fftfreq(Ny, 1/Ny)
kz = cp.fft.fftfreq(Nz, 1/Nz)
KZ, KY, KX = cp.meshgrid(kz, ky, kx, indexing='ij')

k_sq = KX**2 + KY**2 + KZ**2
k_round = cp.round(cp.sqrt(k_sq)).astype(np.int16); k_max = cp.max(k_round)
k_index, k_count = cp.unique(k_round, return_counts = True)
k_f_max_sq = 2
k_forcing = cp.where(k_sq <= k_f_max_sq, k_sq, 0) # k_sq <= 2 / 4 / 8 / 9
indicator = cp.where(k_sq <= k_f_max_sq, 1, 0)
_, count = cp.unique(k_forcing, return_counts = True)
N_forcing = cp.sum(count[1:])
k_f = cp.round(cp.sqrt(k_forcing)).astype(cp.int16)
k_sq[0, 0, 0] = 1  # Avoid division by zero
logging.info(f"Number of forcing components when k_f = {cp.sqrt(k_f_max_sq)}: {N_forcing}")

# Physical parameters
nu = 0.006416 # 1.0e-3  # Kinematic viscosity (adjust as needed)
# rho =



def generate_turbulence_field(n):
    nR = n*n*n; d = L/n

    u0 = np.zeros(nR, dtype = np.float64); v0 = np.zeros(nR, dtype = np.float64); w0 = np.zeros(nR, dtype = np.float64)

    for idxe in range(nR//8):
        ex = 2.0*np.random.randint(2) - 1.0; ey = 2.0*np.random.randint(2) - 1.0; ez = 2.0*np.random.randint(2) - 1.0 # direction vectors
        px = L*np.random.rand(); py = L*np.random.rand(); pz = L*np.random.rand() # random positions
        ip = int(px/d); jp = int(py/d); kp = int(pz/d) # cell position indices

        for k in range(kp-1,kp+3):
            cz = k*d
            for j in range(jp-1,jp+3):
                cy = j*d
                for i in range(ip-1,ip+3):
                    cx = i*d
                    fe = 0.5*(1.0 - 0.5*np.abs(px - cx)/d) * (1.0 - 0.5*np.abs(py - cy)/d) * (1.0 - 0.5*np.abs(pz - cz)/d) # effect weight
                    idxu = (i + n)%n + ((j + n)%n)*n + ((k + n)%n)*n*n
                    u0[idxu] += ex*fe; v0[idxu] += ey*fe; w0[idxu] += ez*fe
    u0 = np.reshape(u0 - u0.mean(), (n,n,n)); v0 = np.reshape(v0 - v0.mean(), (n,n,n)); w0 = np.reshape(w0 - w0.mean(), (n,n,n))
    return u0, v0, w0

def divergence_free(uk, vk, wk):
    continuityk = KX*uk + KY*vk + KZ*wk
    continuity = ifftn(continuityk, axes = (0,1,2), norm = 'forward').real
    return continuity

def zero_cat(X):
    pad = cp.zeros((Mz,My,Mx), dtype = np.complex128)
    X = cp.concatenate((X[:,:,:Nx//2],pad[:Nz,:Ny,Nx//2:-Nx//2],X[:,:,Nx//2:]), axis = 2)
    X = cp.concatenate((X[:,:Ny//2,:],pad[:Nz,Ny//2:-Ny//2,:],X[:,Ny//2:,:]), axis = 1)
    X = cp.concatenate((X[:Nz//2,:,:],pad[Nz//2:-Nz//2,:,:],X[Nz//2:,:,:]), axis = 0)
    return X

def zero_padding(X):
    X[:,:,Nx//2:-Nx//2] = 0; X[:,Ny//2:-Ny//2,:] = 0;
    return X

def removing_interpolation(X):
    X = cp.concatenate((X[:,:,:Nx//2],X[:,:,-Nx//2:]), axis = 2)
    X = cp.concatenate((X[:,:Ny//2,:],X[:,-Ny//2:,:]), axis = 1)
    X = cp.concatenate((X[:Nz//2,:,:],X[-Nz//2:,:,:]), axis = 0)
    return X

def vel_cross_vor(uk, vk, wk):
    uk_pad = zero_cat(uk); u_pad = ifftn(uk_pad, axes = (0,1,2), norm = 'forward').real
    vk_pad = zero_cat(vk); v_pad = ifftn(vk_pad, axes = (0,1,2), norm = 'forward').real
    wk_pad = zero_cat(wk); w_pad = ifftn(wk_pad, axes = (0,1,2), norm = 'forward').real

    # Compute vorticity
    OMExk = 1J*(KY*wk - KZ*vk); OMExk_pad = zero_cat(OMExk)
    OMEyk = 1J*(KZ*uk - KX*wk); OMEyk_pad = zero_cat(OMEyk)
    OMEzk = 1J*(KX*vk - KY*uk); OMEzk_pad = zero_cat(OMEzk)
    OMEx_pad = ifftn(OMExk_pad, axes = (0,1,2), norm = 'forward').real
    OMEy_pad = ifftn(OMEyk_pad, axes = (0,1,2), norm = 'forward').real
    OMEz_pad = ifftn(OMEzk_pad, axes = (0,1,2), norm = 'forward').real

    # Compute nonlinear (rotational) part
    Hu_pad = v_pad*OMEz_pad - w_pad*OMEy_pad; Huk_pad = fftn(Hu_pad, axes = (0,1,2), norm = 'forward')
    Hv_pad = w_pad*OMEx_pad - u_pad*OMEz_pad; Hvk_pad = fftn(Hv_pad, axes = (0,1,2), norm = 'forward')
    Hw_pad = u_pad*OMEy_pad - v_pad*OMEx_pad; Hwk_pad = fftn(Hw_pad, axes = (0,1,2), norm = 'forward')
    Huk = removing_interpolation(Huk_pad)
    Hvk = removing_interpolation(Hvk_pad)
    Hwk = removing_interpolation(Hwk_pad)
    return Huk, Hvk, Hwk

def nonlinear(Huk, Hvk, Hwk):
    pk = -1/k_sq*1J*(KX*Huk + KY*Hvk + KZ*Hwk)
    NLuk, NLvk, NLwk = -1J*KX*pk + Huk, -1J*KY*pk + Hvk, -1J*KZ*pk + Hwk
    return NLuk, NLvk, NLwk, pk

def linear_forcing(uk, vk, wk):
    ufk, vfk, wfk = 0.1*uk, 0.1*vk, 0.1*wk
    mode_E = 0.5*cp.sum(ufk*cp.conjugate(ufk) + vfk*cp.conjugate(vfk) + wfk*cp.conjugate(wfk)).real
    return ufk, vfk, wfk, mode_E

# def TG_forcing():
# omega = 2 * np.pi / L
# forcing = np.array([np.sin(omega * x), np.cos(omega * y), 0])

# def forcing(uk, vk, wk):
#     d = dissipation(uk, vk, wk)
#     ufk, vfk, wfk = indicator*uk, indicator*vk, indicator*wk
#     # coeffu = cp.where(k_sq <= k_f_max_sq, cp.sqrt((ufk*cp.conjugate(ufk)).real), 1);
#     # coeffv = cp.where(k_sq <= k_f_max_sq, cp.sqrt((vfk*cp.conjugate(vfk)).real), 1);
#     # coeffw = cp.where(k_sq <= k_f_max_sq, cp.sqrt((wfk*cp.conjugate(wfk)).real), 1);
#     coeffu = cp.where(k_sq <= k_f_max_sq, cp.abs(ufk), 1); ufk = d*ufk/coeffu
#     coeffv = cp.where(k_sq <= k_f_max_sq, cp.abs(vfk), 1); vfk = d*vfk/coeffv
#     coeffw = cp.where(k_sq <= k_f_max_sq, cp.abs(wfk), 1); wfk = d*wfk/coeffw
#     ufk[0, 0, 0] = vfk[0, 0, 0] = wfk[0, 0, 0] = 0
#
#     mode_E = cp.sum(ufk*cp.conjugate(ufk) + vfk*cp.conjugate(vfk) + wfk*cp.conjugate(wfk)).real
#     return ufk, vfk, wfk, mode_E

def forcing(uk, vk, wk):
    d = dissipation(uk, vk, wk)
    ufk, vfk, wfk = indicator*uk, indicator*vk, indicator*wk
    TKEfk = 0.5*(ufk*cp.conjugate(ufk) + vfk*cp.conjugate(vfk) + wfk*cp.conjugate(wfk)).real
    TKEf = cp.sum(TKEfk).real
    # mode_E = (ufk*cp.conjugate(ufk) + vfk*cp.conjugate(vfk) + wfk*cp.conjugate(wfk)).real
    # mode_E = cp.where(k_sq <= k_f_max_sq, mode_E, 1)

    fuk, fvk, fwk = d/(2*TKEf)*ufk, d/(2*TKEf)*vfk, d/(2*TKEf)*wfk
    fuk[0, 0, 0] = fvk[0, 0, 0] = fwk[0, 0, 0] = 0
    forcing_TKE = cp.sum(0.5*(fuk*cp.conjugate(fuk) + fvk*cp.conjugate(fvk) + fwk*cp.conjugate(fwk)).real)
    return fuk, fvk, fwk, TKEf, forcing_TKE

def diff_exp(t):
    return exp(nu*k_sq*t)

def e_spectrum(uk, vk, wk):
    Eu_3D, Ev_3D, Ew_3D = 2*pi*k_sq*uk*cp.conjugate(uk), 2*pi*k_sq*vk*cp.conjugate(vk), 2*pi*k_sq*wk*cp.conjugate(wk)
    spectrum = cp.zeros_like(k_index, dtype = np.float64)
    for i in k_index:
        spectrum[i] = cp.sum(Eu_3D.real[i == k_round]) + cp.sum(Ev_3D.real[i == k_round]) + cp.sum(Ew_3D.real[i == k_round])
    return spectrum

def dissipation(uk, vk, wk): # 2*nu*<S_{ij}S_{IJ}>
    u_xk, u_yk, u_zk = 1J*KX*uk, 1J*KY*uk, 1J*KZ*uk
    v_xk, v_yk, v_zk = 1J*KX*vk, 1J*KY*vk, 1J*KZ*vk
    w_xk, w_yk, w_zk = 1J*KX*wk, 1J*KY*wk, 1J*KZ*wk
    u_x, u_y, u_z = ifftn(u_xk, axes = (0,1,2), norm = 'forward').real, ifftn(u_yk, axes = (0,1,2), norm = 'forward').real, ifftn(u_zk, axes = (0,1,2), norm = 'forward').real
    v_x, v_y, v_z = ifftn(v_xk, axes = (0,1,2), norm = 'forward').real, ifftn(v_yk, axes = (0,1,2), norm = 'forward').real, ifftn(v_zk, axes = (0,1,2), norm = 'forward').real
    w_x, w_y, w_z = ifftn(w_xk, axes = (0,1,2), norm = 'forward').real, ifftn(w_yk, axes = (0,1,2), norm = 'forward').real, ifftn(w_zk, axes = (0,1,2), norm = 'forward').real

    d = 2*nu*cp.mean(u_x**2 + v_y**2 + w_z**2 + 2*((0.5*(u_y + v_x))**2 + (0.5*(v_z + w_y))**2 + (0.5*(w_x + u_z))**2))
    return d

def rms_vel_mag(u, v, w):
    u_rms, v_rms, w_rms = np.std(u), np.std(v), np.std(w)
    return np.sqrt((u_rms**2 + v_rms**2 + w_rms**2)/3)

def taylor_micro_scale(U_rms, d):
    return np.sqrt(15*nu*U_rms**2/d)

def reynolds_num(U_rms, lambda_g):
    return U_rms*lambda_g/nu

def kolmogorov_length_scale(d):
    return (nu**3/d)**(1/4)

def kolmogorov_time_scale(d):
    return np.sqrt(nu/d)



# Initial field generation
logging.info(f"Initial condition using random vortex method")
u_init, v_init, w_init = generate_turbulence_field(N); U_rms = rms_vel_mag(u_init, v_init, w_init) # In CPU

# Initial condition and spectrum manipulation
u, v, w = cp.asarray(u_init), cp.asarray(v_init), cp.asarray(w_init)
uk, vk, wk = fftn(u, axes = (0,1,2), norm = 'forward'), fftn(v, axes = (0,1,2), norm = 'forward'), fftn(w, axes = (0,1,2), norm = 'forward')
# init_coeff = cp.asarray(U_rms)**2/cp.sqrt(k_f_max_sq)*cp.where(k_sq <= 2, k_sq/k_f_max_sq, cp.sqrt(k_sq/k_f_max_sq)**(-5/3))
# uk, vk, wk = init_coeff*uk/(cp.abs(uk) + 10**-8), init_coeff*vk/(cp.abs(vk) + 10**-8), init_coeff*wk/(cp.abs(wk) + 10**-8)
# uk[0, 0, 0] = vk[0, 0, 0] = wk[0, 0, 0] = 0

u = ifftn(uk, axes = (0,1,2), norm = 'forward').real; us = np.expand_dims(cp.asnumpy(u), axis = (0,1))
v = ifftn(vk, axes = (0,1,2), norm = 'forward').real; vs = np.expand_dims(cp.asnumpy(v), axis = (0,1))
w = ifftn(wk, axes = (0,1,2), norm = 'forward').real; ws = np.expand_dims(cp.asnumpy(w), axis = (0,1))
vel = np.concatenate((us,vs,ws), axis = 1)

CreateData = os.path.join(Train_Dir, f"Train{1} step{0} 3D HIT n={N} T={T} dt={dt:.4f} K={K} data.h5")
fc = h5py.File(CreateData, 'w')
fc.create_dataset(f"vel{0}", data = vel)
fc.close()

# Initial statistics
U_rms = rms_vel_mag(us[0,0,:], vs[0,0,:], ws[0,0,:])
TKE, eddy_turnover = 3/2*U_rms**2, L/U_rms
d = cp.asnumpy(dissipation(uk, vk, wk))
lambda_g, eta, tau_eta = taylor_micro_scale(U_rms, d), kolmogorov_length_scale(d), kolmogorov_time_scale(d)
Re = reynolds_num(U_rms, lambda_g)
E_spectrum = cp.asnumpy(e_spectrum(uk, vk, wk))
_, _, _, TKE_f_modes, forcing_E = forcing(uk, vk, wk); TKE_f_modes, forcing_E = TKE_f_modes.get(), forcing_E.get()

logging.info(f"Initial U_rms = {U_rms:.10f}, TKE_f_modes = {TKE_f_modes:.10f}, coeff = {d**2/(4*TKE_f_modes):.10f}:.10f, forcing E = {forcing_E:.10f}, dissipation = {d:.10f}")

Filewrite1 = os.path.join(Train_Dir, f"Train{1} statistics.plt")
fstat = open(Filewrite1, 'w')
fstat.write('VARIABLES="t","U_rms","TKE","forcing E","dissipation","Taylor scale","Re","Kolmo length","Kolmo time","eddy turnover"\n')
fstat.write(f'Zone T="T=Train{1}"\n')
fstat.write(f'{dt*0:.4f} {U_rms} {TKE} {forcing_E} {d} {lambda_g} {Re} {eta} {tau_eta} {eddy_turnover}\n')
fstat.close()

Filewrite2 = os.path.join(Train_Dir, f"Train{1} energy spectrum.plt")
fE = open(Filewrite2, 'w')
fE.write('VARIABLES="k","E"\n')
fE.write(f'Zone T="T={dt*0:.4f}"\n')
for i in range(N//2+1):
    fE.write('%d %.10lf\n' %(k_index.get()[i],E_spectrum[i]))
fE.close()

# RK3 coefficients
RKa = cp.zeros(3, dtype = np.float64); RKa[0], RKa[1], RKa[2] = 8.0/15, 5.0/12, 3.0/4
RKb = cp.zeros(3, dtype = np.float64); RKb[0], RKb[1], RKb[2] = 0.0, -17.0/60, -5.0/12
RKc = cp.zeros(4, dtype = np.float64); RKc[0], RKc[1], RKc[2], RKc[3] = 0.0, 8.0/15, 2.0/3, 1.0

NLxk_p, NLyk_p, NLzk_p = cp.zeros_like(uk), cp.zeros_like(vk), cp.zeros_like(wk)
fxk_p, fyk_p, fzk_p = cp.zeros_like(uk), cp.zeros_like(vk), cp.zeros_like(wk)
logging.info(f"Starting time advancement using RK3")
# logging.info(f"Starting simulation {num} using RK3")
# Time-stepping loop
for q in range(1,Nt+1):  # Adjust the number of time steps as needed
    for s in range(3): # RK3 iteration
        Hxk, Hyk, Hzk = vel_cross_vor(uk, vk, wk)
        NLxk, NLyk, NLzk, _ = nonlinear(Hxk, Hyk, Hzk)
        fxk, fyk, fzk, _, _ = forcing(uk, vk, wk)

        uk = (RKa[s]*dt*(NLxk + fxk) + uk)*diff_exp((RKc[s] - RKc[s+1])*dt) + RKb[s]*dt*(NLxk_p + fxk_p)*diff_exp((RKc[s-1] - RKc[s+1])*dt)
        vk = (RKa[s]*dt*(NLyk + fyk) + vk)*diff_exp((RKc[s] - RKc[s+1])*dt) + RKb[s]*dt*(NLyk_p + fyk_p)*diff_exp((RKc[s-1] - RKc[s+1])*dt)
        wk = (RKa[s]*dt*(NLzk + fzk) + wk)*diff_exp((RKc[s] - RKc[s+1])*dt) + RKb[s]*dt*(NLzk_p + fzk_p)*diff_exp((RKc[s-1] - RKc[s+1])*dt)
        uk[0, 0, 0] = vk[0, 0, 0] = wk[0, 0, 0] = 0

        NLxk_p[:], NLyk_p[:], NLzk_p[:], fxk_p[:], fyk_p[:], fzk_p[:] = NLxk[:], NLyk[:], NLzk[:], fxk[:], fyk[:], fzk[:]

    if q % K == 0:
        u = ifftn(uk, axes = (0,1,2), norm = 'forward').real; us = np.expand_dims(cp.asnumpy(u), axis = (0,1))
        v = ifftn(vk, axes = (0,1,2), norm = 'forward').real; vs = np.expand_dims(cp.asnumpy(v), axis = (0,1))
        w = ifftn(wk, axes = (0,1,2), norm = 'forward').real; ws = np.expand_dims(cp.asnumpy(w), axis = (0,1))
        vel = np.concatenate((us,vs,ws), axis = 1)

        CreateData = os.path.join(Train_Dir, f"Train{1} step{q//K} 3D HIT n={N} T={T} dt={dt:.4f} K={K} data.h5")
        fc = h5py.File(CreateData, 'w')
        fc.create_dataset(f"vel{q//K}", data = vel)
        fc.close()

        U_rms = rms_vel_mag(us[0,0,:], vs[0,0,:], ws[0,0,:])
        TKE, eddy_turnover = 3/2*U_rms**2, L/U_rms
        d = cp.asnumpy(dissipation(uk, vk, wk))
        lambda_g, eta, tau_eta = taylor_micro_scale(U_rms, d), kolmogorov_length_scale(d), kolmogorov_time_scale(d)
        Re = reynolds_num(U_rms, lambda_g)
        E_spectrum = cp.asnumpy(e_spectrum(uk, vk, wk))
        _, _, _, TKE_f_modes, forcing_E = forcing(uk, vk, wk); TKE_f_modes, forcing_E = TKE_f_modes.get(), forcing_E.get()

        fstat = open(Filewrite1, 'a')
        fstat.write(f'{dt*q:.4f} {U_rms} {TKE} {forcing_E} {d} {lambda_g} {Re} {eta} {tau_eta} {eddy_turnover}\n')
        fstat.close()

        fE = open(Filewrite2, 'a')
        fE.write(f'Zone T="T={dt*q:.4f}"\n')
        for i in range(N//2+1):
            fE.write('%d %.10lf\n' %(k_index.get()[i],E_spectrum[i]))
        fE.close()

    if q % (10*K) == 0: logging.info(f"{q} time marching is done. U_rms = {U_rms:.10f}, TKE_f_modes = {TKE_f_modes:.10f}, coeff = {d**2/(4*TKE_f_modes):.10f}:.10f, forcing E = {forcing_E:.10f}, dissipation = {d:.10f}")

# logging.info(f"Simulation {num} is done.")
