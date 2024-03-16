#statistics.py
def dissipation(uk, vk): # 2*nu*<S_{ij}S_{IJ}>
    u_xk,u_yk = 1J*KX*uk, 1J*KY*uk
    v_xk,v_yk = 1J*KX*vk, 1J*KY*vk
    u_x,u_y= irfftn(u_xk, axes = (0,1), norm = 'forward').real,irfftn(u_yk, axes = (0,1), norm = 'forward').real
    v_x,v_y= irfftn(v_xk, axes = (0,1), norm = 'forward').real,irfftn(v_yk, axes = (0,1), norm = 'forward').real
    d = 2*nu*cp.mean(u_x**2 + v_y**2 + 2*((0.5*(u_y + v_x))**2))
    return d

def kolmogorov_length_scale(d):
    return (nu**3/d)**(1/4)

def kolmogorov_time_scale(d):
    return cp.sqrt(nu/d)


def e_spectrum(uk, vk, wk):
    Eu_3D, Ev_3D, Ew_3D = 2*pi*k_sq*uk*cp.conjugate(uk), 2*pi*k_sq*vk*cp.conjugate(vk), 2*pi*k_sq*wk*cp.conjugate(wk)
    spectrum = cp.zeros_like(k_index, dtype = np.float64)
    for i in k_index:
        spectrum[i] = cp.sum(Eu_3D.real[i == k_round]) + cp.sum(Ev_3D.real[i == k_round]) + cp.sum(Ew_3D.real[i == k_round])
    return spectrum

def rms_vel_mag(u, v, w):
    u_rms, v_rms, w_rms = np.std(u), np.std(v), np.std(w)
    return np.sqrt((u_rms**2 + v_rms**2 + w_rms**2)/3)

def taylor_micro_scale(U_rms, d):
    return np.sqrt(15*nu*U_rms**2/d)

def reynolds_num(U_rms, lambda_g):
    return U_rms*lambda_g/nu