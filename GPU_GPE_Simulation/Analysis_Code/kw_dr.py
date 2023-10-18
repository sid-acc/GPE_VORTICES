import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy.fft import fft, fft2, fftfreq, fftshift
from mpl_toolkits.mplot3d import Axes3D 
from scipy.optimize import curve_fit
import cmath
import os
import scipy.special
import warnings
from scipy.optimize import OptimizeWarning
from matplotlib.animation import FuncAnimation

import helpers

hbar = 1.05457 *10**(-34)
hbar2 = 1.054 *10**(-25)
hbar3 = 1.05457 *10**(-31)
hbar4 = 1.054 *10**(-22)
kb =  1.38*(10**(-32))
kb2 = 1.38*(10**(-26))
mass = 12 * 1.66 * 10**(-27)
a0 = 5.29 * 10**(-5)
timestep = 5.0e-3


def xi_fit(directory, NX, NY, NZ, Xrange, vortex):
    '''
    Fits vertical profile of density to xi function and extracts xi value
    '''
    slices = helpers.get_data(directory + '\\I_dens.t_+0.000e+00.csv', NX, NY, NZ, False)
    X = np.linspace(-(Xrange/2), Xrange/2, NX)
    midline = slices[(int) (NZ/2), (int) (NY/2)]
    sizes = np.full(NX, 10)

    if (not vortex):
        popt, pcov = curve_fit(helpers.f, X[(int) (NX/2) + 12:], midline[12:(int) (NX/2)])
        plt.scatter(X, midline, sizes)
        plt.plot(X[(int) (NX/2) + 12:] - Xrange/2, helpers.f(X[(int) (NX/2) + 12:], *popt), color='orange')
        print(popt)
        plt.show()
    else:
        popt, pcov = curve_fit(helpers.f, X[(int) (NX/2) + 12:NX-35], midline[12:(int) (NX/2 - 35)])
        popt2, pcov2 = curve_fit(helpers.f, X[(int) (NX/2 - 18):(int) (NX/2)], midline[(int) (NX/2 - 18):(int) (NX/2)])

        #plt.scatter(X, midline, sizes)
        #plt.plot(X[(int) (NX/2) + 12:NX-35] - Xrange/2, helpers.f(X[(int) (NX/2) + 12:NX-35], *popt), color='orange')
        #plt.plot(X[(int) (NX/2 - 18):(int) (NX/2)], helpers.f(X[(int) (NX/2 - 18):(int) (NX/2)], *popt2), color='green')
        
        #print(f'n ; {popt[0]}')
        #plt.show()
        print(popt2[2])
        return popt2[2]
        # (popt[2] + popt2[2])*(np.sqrt(2))/2 #xi = hbar/sqrt(gn)

def energy(dens, Kdens, K2, td_pot, g, dtau, dktau, NX, NY, NZ):
        '''
        Calculates energy from 2D array inputs
        '''
        ke = np.sum(dktau*(K2*Kdens))
        pe = np.sum(dtau*(td_pot*dens))
        int_e = np.sum((dens*dens)*dtau)
        N = np.sum(dtau*dens)
        print(N)
        ke = -ke*NX*NY*NZ*(hbar2/(N*kb2*timestep))
        pe = pe*(1/N)
        int_e = int_e*(g/(2*kb2*N))

        return ke, pe, int_e

    
def v_energy_over_time(directory, NX, NY, NZ, Xrange, Yrange, Zrange, a_s, n, start, stop, step):
    '''
    Calculates and plots energy over time for a data folder
    '''
    dtau = (Xrange/NX)*(Yrange/NY)*(Zrange/NZ) 
    dktau = 8*(np.pi**3)/(Xrange*Yrange*Zrange)
    g = 4 * np.pi * a0 * a_s *(hbar2**2)/mass
    #print(g*n/kb2)

    K2_filename = f'{directory}\\I_K2.t.csv'
    K2 = np.array(helpers.get_data(K2_filename, NX, NY, NZ, False))

    ke_list = []
    pe_list = []
    int_e_list = []

    ke_list2 = []
    pe_list2 = []
    int_e_list2 = []
    
    times1, times2 = helpers.times_array(start, stop, step)
    for i, t in enumerate(times1):
        
        dens_filename = f'{directory}\\I_dens.t_+{t}.csv'
        Kdens_filename = f'{directory}\\I_Kdens.t_+{t}.csv'
        td_pot_filename = f'{directory}\\I_td_pot.t_+{t}.csv'
       
        dens = np.array(helpers.get_data(dens_filename, NX, NY, NZ, False))
        Kdens = np.array(helpers.get_data(Kdens_filename, NX, NY, NZ, False))
        td_pot = np.array(helpers.get_data(td_pot_filename, NX, NY, NZ, True))

        dens2 = np.array(helpers.get_data_radius(dens, Xrange, Yrange, NX, NY, NZ, 5))
        Kdens2 = np.array(helpers.get_data_radius(Kdens, Xrange, Yrange, NX, NY, NZ, 5))
        td_pot2 = np.array(helpers.get_data_radius(td_pot, Xrange, Yrange, NX, NY, NZ, 5))
        
        ke, pe, int_e = energy(dens, Kdens, K2, td_pot, g, dtau, dktau, NX, NY, NZ)
        ke2, pe2, int_e2 = energy(dens2, Kdens2, K2, td_pot2, g, dtau, dktau)
        ke_list.append(ke)
        pe_list.append(pe)
        int_e_list.append(int_e)

        ke_list2.append(ke2)
        pe_list2.append(pe2)
        int_e_list2.append(int_e2)


    helpers.plot_energy(times2, ke_list, int_e_list)
    helpers.plot_energy(times2, ke_list2, int_e_list2)

    return ke_list, pe_list, int_e_list, ke_list2, pe_list2, int_e_list2
    
def get_mu(directory, NX, NY, NZ, Xrange, Yrange, Zrange, a_s, n):
    '''
    Explicitly calculates mu via Pethick book ch. 6 definition
    mu_xi_def is better to use
    '''
    dtau = (Xrange/NX)*(Yrange/NY)*(Zrange/NZ) 
    dktau = 8*(np.pi**3)/(Xrange*Yrange*Zrange)
    g = 4 * np.pi * a0 * a_s *(hbar2**2)/mass

    K2_filename = f'{directory}\\I_K2.t.csv'
    K2 = np.array(helpers.get_data(K2_filename, NX, NY, NZ, False))
    z = '0.000e+00'
    dens_filename = f'{directory}\\I_dens.t_+{z}.csv'
    Kdens_filename = f'{directory}\\I_Kdens.t_+{z}.csv'
    td_pot_filename = f'{directory}\\I_td_pot.t_+{z}.csv'
    
    dens = np.array(helpers.get_data(dens_filename, NX, NY, NZ, False))
    Kdens = np.array(helpers.get_data(Kdens_filename, NX, NY, NZ, False))
    td_pot = np.array(helpers.get_data(td_pot_filename, NX, NY, NZ, True))

    ke, pe, int_e = energy(dens, Kdens, K2, td_pot, g, dtau, dktau, NX, NY, NZ)
    return -ke + pe + (2*int_e)

def vortex_core(phase, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange):
    '''
    Gets x, y, z coordinates of vortex core using phase singularity
    '''
    Z_start_idx =((int) (NZ/2)) - ((int) ((height/2)/(Zrange/NZ)))
    Z_end_idx = NZ - 1 - Z_start_idx
    Y_start_idx = ((int) (NY/2)) - ((int) ((radius)/(Yrange/NY)))
    Y_end_idx = NY - 1 - Y_start_idx
    X_start_idx = ((int) (NX/2)) - ((int) ((radius)/(Xrange/NX)))
    X_end_idx = NX - 1 - X_start_idx

    #print(Z_start_idx, Z_end_idx)
    core_x = []
    core_y = []
    core_z = []
    for i in range(Z_start_idx, Z_end_idx):
        zeros_x = []
        zeros_y = []
        for jx in range (X_start_idx, X_end_idx):
            for k in range(Y_start_idx, Y_end_idx):
                #print(i, jx, k)
                if phase[i, k, jx] == 0 and abs(jx - (int)(NX/2)) < radius/(2*Xrange/NX) and abs(k - (int)(NY/2)) < radius/(2*Yrange/NY):
                    zeros_x.append(jx)
                    zeros_y.append(k)
        if len(zeros_x) == 0 or len(zeros_y) == 0:
            #print('here')
            zeros_x.append(core_x[len(core_x) - 1])
            zeros_y.append(core_y[len(core_y) - 1])
        core_z.append(i)
        core_x.append(zeros_x[(int)(len(zeros_x)/2)])
        core_y.append(zeros_y[(int)(len(zeros_y)/2)])
            
        X_start_idx = min(zeros_x) - 2
        X_end_idx = max(zeros_x) + 2
        Y_start_idx = min(zeros_y) - 2
        Y_end_idx = max(zeros_y) + 2
    
    X = np.linspace(-(Xrange/2), Xrange/2, NX)
    Y = np.linspace(-(Yrange/2), Yrange/2, NX)
    Z = np.linspace(-(Zrange/2), Zrange/2, NZ)
     
    #plt.plot(X[core_x], Z[core_z])
    #plt.plot(Y[core_y], Z[core_z])
    #plt.xlim(-Xrange/2, Xrange/2)
    #plt.ylim(-Zrange/2, Zrange/2)
    #plt.show()
    
    return (X[core_x], Y[core_y], Z[core_z])

def vc_length(x_t, y_t, z_t):
    '''
    Calculates length of vortex core given x, y , z coordinates
    '''
    length = 0
    for i, z in enumerate(z_t[:len(z_t)-1]):
        length += np.sqrt((z_t[i+1] - z)**2 + (x_t[i+1]-x_t[i])**2 + (y_t[i+1]-y_t[i])**2)
    return length

def vc_data(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times, dir):
    '''
    Gets all vortex cores over time interval 'times', returns 2D arrays x, y, and 1D array z
    '''
    x = []
    y = []
    for i, t in enumerate(times):
        #print(t)
        phase_filename = f'{dir}\\I_phase.t_+{t}.csv'
        phase = helpers.get_data(phase_filename, NX, NY, NZ, False)
        helpers.plot_slices(phase, NX, NY, NZ, Xrange, Yrange, Zrange)
        x_t, y_t, z_t = vortex_core(phase, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange)
        x.append(x_t)
        y.append(y_t)
    
    return x, y, z_t

def vortex_core_animation(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times, dir):
   
    fig, ax = plt.subplots()
    x, y, z = vc_data(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times, dir)
    def plot_vc_core(i):
        ax.clear()
        ax.plot(x[i], z)
        ax.plot(y[i], z)
        ax.set_xlim(-Xrange/2, Xrange/2)
        ax.set_ylim(-Xrange/2, Xrange/2)
        ax.set_title(f'{times[i]} ms')
        ax.set_xlabel('x, y (um)')
        ax.set_ylabel('z (um)')
        ax.legend(['x(z)', 'y(z)'])
        
    ani = FuncAnimation(fig, plot_vc_core, frames=len(times), interval=500, repeat=True)
    
    plt.show()

def get_max_kz(kz, omega, w2d, w1d, nodes, times):
    '''
    Gets amplitude over time or over frequency space of desired KW mode
    '''
    kz_idx1 = np.argmin(abs(nodes-kz))
    kz_idx2 = np.argmin(abs(-nodes-kz))
    #plt.plot(times,w1d[:, kz_idx1])
    #plt.plot(times,w1d[:, kz_idx2])
    #plt.show()
    r = (np.mean(w1d[:,kz_idx1]), np.std(w1d[:,kz_idx1]))
    l = (np.mean(w1d[:, kz_idx2]), np.std(w1d[:, kz_idx2]))
    if r[0] > l[0]:
        return (r, w2d[:,kz_idx1])
    else:
        return (l, w2d[:,kz_idx2])

def extract_max(kz, omega, w2d, w1d, nodes, lorentz, times):
    max_tup = get_max_kz(kz, omega, w2d, w1d, nodes, times)
    if not lorentz:
        #line = max_tup[1]
        #popt, pcov = curve_fit(helpers.f1, times, line)
        return max_tup[0]#(popt[0], np.sqrt(pcov[0][0]))
    else: 
        line = max_tup[1]
        popt, pcov= helpers.lorentzian_fit(omega, line)
        print(popt[0]/(np.sqrt(popt[2])))
        return (popt[0]/(np.sqrt(popt[2])), np.sqrt((pcov[0][0]/(popt[0]**2)) + (pcov[2][2]/(popt[2]**2))))
    
def vc_sin_fit(core_x, core_y, z, radius, times, nodes, length):
    '''
    Gets amplitude of KW by fitting to sin function
    '''
    amps = []
    errs = []
    for i,c in enumerate(core_x):
        z_new = z[6:len(z)-6]
        c_x = np.array(c[6:len(c)-6])/radius
        c_y = np.array(core_y[i][6:len(core_y[i])-6])/radius
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeError)
            try: 
                popt_x, pcov_x = curve_fit(helpers.sin_func, z_new, c_x, p0 = [0.03, 0, nodes*np.pi/length, 0])
                popt_y, pcov_y = curve_fit(helpers.sin_func, z_new, c_y, p0 = [0.03, 0, nodes*np.pi/length, 0])
                #plt.plot(z_new, c_x)
                #plt.plot(np.linspace(min(z_new), max(z_new), 100), helpers.sin_func(np.linspace(min(z_new), max(z_new), 100), *popt_x))
                #plt.title("X")
                #plt.show()
                #plt.plot(z_new, c_y)
                #plt.plot(np.linspace(min(z_new), max(z_new), 100), helpers.sin_func(np.linspace(min(z_new), max(z_new), 100), *popt_y))
                #plt.title("Y")
                #plt.show()
                amp = np.sqrt(popt_x[0]**2 + popt_y[0]**2)
                err = np.sqrt(pcov_x[0][0]**2 + pcov_y[0][0]**2)
                if (err/amp < 0.1):
                    amps.append(amp)
                    errs.append(err)
                    print('added')
                else:
                    amps.append(0)
                    errs.append(0)
            except RuntimeError:
                amps.append(0)
                errs.append(0)
    max_amp = max(amps)
    popt, pcov = curve_fit(helpers.f1, times, amps)
    #plt.scatter(times, amps)
    #plt.plot(times, helpers.f1(times, *popt))
    #plt.show()
    return (helpers.f1((times[len(times)-1] - times[0])/2, *popt), np.sqrt(pcov[0][0]),amps, errs)
    

def gaussian_fit(exp_mean, sweep, freqs, amps):
    '''
    Fits KW spectrum to Gaussian
    '''
    with warnings.catch_warnings():
        idx = 0
        while True:
            warnings.simplefilter("error", OptimizeWarning)
            warnings.simplefilter("error", RuntimeWarning)
            idy = (int) (idx/sweep)
            idz = (int) (idy/sweep)
            p0_2 = np.linspace(1/(2*(100**2)),1/(2*(1)), sweep)
            p0_3 = np.linspace(0, 5 ,sweep)
            p0_4 = np.linspace(-5, 5, sweep)
            if (idx > sweep**3):
                break
            try: 
                #print(idx)
                #print(p0_2[idx%sweep], p0_3[idy%sweep], p0_4[idz%sweep])
                #popt, pcov = curve_fit(helpers.gaussian, freqs, amps)
                #print(popt)
                popt, pcov = curve_fit(helpers.gaussian, freqs, amps, p0=[exp_mean, p0_2[idx%sweep], p0_3[idy%sweep], p0_4[idz%sweep]])
                #print(popt[0], np.sqrt(pcov[0][0]))
                if (np.sqrt(pcov[0][0]) < popt[0]/5  and popt[2] > 0):
                    return (popt, pcov)
                idx += 1
            except RuntimeError: 
                #print('error')
                idx += 1
            except OptimizeWarning:
                #print('error')
                idx +=1
            except RuntimeWarning:
                #print('error')
                idx += 1
            
def get_lengths(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, dir, start, stop, step):
    '''
    Gets lengths of vortex cores over time interval 
    '''
    lengths = []
    times1, _ = helpers.times_array(start, stop, step)
    x, y, z = vc_data(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times1, dir)
    for i, x_t in enumerate(x):
        lengths.append(vc_length(x_t, y[i], z))
    #plt.scatter(times2, lengths)
    #plt.show()
    return np.array(lengths)-lengths[0]


def vc_ffts(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, dir, start, stop, step):
    '''
    Gets vortex core data over time interval and calculates fourier transforms
    '''
    times1, times2 = helpers.times_array(start, stop, step)
    x, y, z = vc_data(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times1, dir)
    w =  (np.array(x) + 1j *np.array(y))/radius#np.array(x)# + 1j *
    
    kz, omega, wf2d, wf1d = helpers.general_fft(times2, z, w)

    return (z, kz, times2, omega, wf2d, wf1d)

def kw_spectrum(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, n, dir, lorentz, start, stop, step):
    '''
    Gets vortex core data over time interval and plots fourier transforms
    '''
    
    z, kz, times2, omega, wf2d, wf1d = vc_ffts(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, dir, start, stop, step)
    helpers.plot_fft(kz, times2, wf1d)
    helpers.plot_fft(kz, omega, wf2d)

    return extract_max(kz, omega, wf2d, wf1d, n, lorentz, times2)

def freq_amp_plot(directory, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, n, exp_mean, start, stop, step, ax):
    '''
    Plots KW spectrum for particular mode. Can change between obtaining amplitude by sin fit or fourier transform.
    Writes data into file
    '''
    #directory = f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\kw_{n}m_{height}l'
    freqs = []
    amps = []
    stds = []
    stength = 0
    
    for sub_dir in os.scandir(directory):
        print(sub_dir.path)
        print(helpers.get_input_param(sub_dir.path, '-ShakeFrequency'))
        strength = helpers.get_input_param(sub_dir.path, '-ShakeAmplitude')
        #print(freqs)
        #val = kw_spectrum(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, n, sub_dir.path, False, start, stop, step)
        times1, times2 = helpers.times_array(start, stop, step)
        x, y, z = vc_data(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times1, sub_dir.path)
        val = vc_sin_fit(x, y, z, radius, times2, n, height)
        #plt.plot(times2, val[2])
        #plt.show()
        amps.append(np.mean(val[2]))
        stds.append(np.std(val[2]))
        freqs.append(helpers.get_input_param(sub_dir.path, '-ShakeFrequency'))
        
    
    print(freqs, amps)
    plt.scatter(freqs, amps)
    plt.show()

    #data = ''  
    #for i, freq in enumerate(freqs):
    #   data += f'{{{freq}, {amps[i]}}},'

    popt, pcov = gaussian_fit(exp_mean, 80, freqs, amps)
    print(popt)  
    width, width_err = helpers.gaussian_width(popt, pcov) 
    print(width)      
    #fig, ax = plt.subplots()
    freqs2 = np.linspace(min(freqs), max(freqs), 50)
    plt.figure(figsize=(6.4,4.8))
    color = 'blue'
    if strength == 0.8:
        color = 'orange'
    
    plt.scatter(freqs, amps, color=color)
    plt.errorbar(freqs, amps, yerr=stds, ls='none', color=color)
    plt.xlabel('w (Hz)')
    plt.ylabel('A/R')
    plt.plot(freqs2, helpers.gaussian(freqs2, *popt), color=color, linestyle='dashed')
    plt.show()

    with open(f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\kw_dr3\\Peaks\\peaks.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([n, height, popt[0], np.sqrt(pcov[0][0])])

    with open(f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\kw_dr3\\Widths\\widths.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([n, height, width, width_err])

    return 


def get_adb_freq(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, nodes, start, stop, step):
    dir = f'V:\\VortexSimulationData\\kw_energy_nresp\\gpeSolver_RK4-2023-08-25T10-03-49'
    _, kz, times2, omega, wf2d, wf1d = vc_ffts(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, dir, start, stop, step)
    plt.pcolormesh(kz, times2, wf1d)
    plt.colorbar()
    plt.show()

    plt.pcolormesh(kz, omega, wf2d)
    plt.colorbar()
    plt.show()

    line = get_max_kz(kz, omega, wf2d, wf1d, nodes, times2)[1]
    popt, pcov= helpers.lorentzian_fit(omega, line)
    print(popt[1], np.sqrt(pcov[1][1]))
    return popt[1], np.sqrt(pcov[1][1])

def monopole_moment(dens_data, height, NZ, Zrange):
    mm = []
    z_array = np.linspace(-Zrange/2, Zrange/2, NZ)
    z_array_new = []
    for i, z in enumerate(dens_data):
        if abs(z_array[i]) <= height/2:
            mm.append(np.sum(z))
            z_array_new.append(z_array[i])
    #plt.plot(z_array_new, mm)
    #plt.show()
    return z_array_new, mm

def dipole_moment(dens_data, height, NY, NZ, Yrange, Zrange):
    dm = []
    y_array = np.linspace(-Yrange/2, Yrange/2, NY)
    z_array = np.linspace(-Zrange/2, Zrange/2, NZ)
    for i, z in enumerate(dens_data):
        sum = 0
        if abs(z_array[i]) <= height/2:
            for i, y in enumerate(z):
                sum += y_array[i]*np.sum(y)
            dm.append(sum)          
    return dm

def get_moments(times_array, directory, height, NX, NY, NZ, Yrange, Zrange):
    mm_list = []
    dm_list = []
    z_array = []
    mm0 = []
    dm0 = 0
    for i, time in enumerate(times_array):
        filename = directory + f'\\I_dens.t_+{time}.csv'
        dens_data = helpers.get_data(filename, NX, NY, NZ, False)
        z_array, mm = monopole_moment(dens_data,height, NZ, Zrange)
        #dm = dipole_moment(dens_data, height, NY, NZ, Yrange, Zrange)
        if i == 0:
            mm0 = mm
            #dm0 = dm
        mm_list.append(np.array(mm) - np.array(mm0))
        #.plot(z_array, mm_list[i])
        #plt.show()
        #dm_list.append(np.array(dm) - np.array(dm0))
    return z_array, mm_list#, dm_list

def phonon_fft(nodes, length, NX, NY, NZ, Xrange, Yrange, Zrange, start, stop, step):
    times1, times2 = helpers.times_array(start, stop, step)
    z_array = []
    directory = f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\dir\\gpeSolver_RK4-2023-09-22T20-49-00'
    for sub_dir in os.scandir(directory):
        z_array, mm_list, dm_list = get_moments(times1, directory, length, NX, NY, NZ, Yrange, Zrange)
        kz, omega, mm_ft2d, mm_ft1d = helpers.general_fft(times2, z_array, mm_list)
        _, _, dm_ft2d, dm_ft1d = helpers.general_fft(times2, z_array, dm_list)
        #helpers.plot_fft(kz, omega, mm_ft2d)
        #get_max_kz(kz, omega, mm_ft2d, mm_ft1d, nodes, times2)
        helpers.plot_fft(kz, times2, mm_ft1d)
        helpers.plot_fft(kz, omega, mm_ft2d)
        get_max_kz(kz, omega, dm_ft2d, dm_ft1d, 1, times2)
        

def gen_exp_data(nodes, radius, height, freq, start, stop, step):
    _, times = helpers.times_array(start, stop, step)
    z = np.linspace(-height/2, height/2, 81)
    data = []
    for t in times:
        core_x = 2*radius*np.sin(((nodes)*np.pi/(height))*z)*np.sin(2*np.pi*freq*0.001*t)
        core_y = -2*radius*np.sin(((nodes)*np.pi/(height))*z)*np.cos(2*np.pi*freq*0.001*t)
        #plt.plot(z, core_x)
        #plt.plot(z, core_y)
        #plt.show()
        data.append(core_x + 1j*core_y)
    kz, omega, ft2d, ft1d = helpers.general_fft(times, z, data)
    helpers.plot_fft(kz, times, ft1d)
    get_max_kz(kz, omega, ft2d, ft1d, nodes, times)

def kw_energy_plot(directory, NX, NY, NZ, Xrange, Yrange, Zrange, a_s, start, stop, step, p_stop):
    energies = []
    energies2 = []

    energies_comp = []
    energies2_comp = []
    kw = []
    _, times = helpers.times_array(start, stop, step)
    for sub_dir in os.scandir(directory):
        print(sub_dir.path)
        length = helpers.get_input_param(sub_dir.path, '-Length')
        nodes = 2*(helpers.get_input_param(sub_dir.path, '-PotentialDivisions'))-1
        kw.append(helpers.get_input_param(sub_dir.path, '-ShakeAmplitude'))
        ke, _, int_e, ke2, _, int_e2 = energy(sub_dir.path, NX, NY, NZ, Xrange, Yrange, Zrange, a_s, 0, start, stop, step)
        total_energy = np.array(ke) + np.array(int_e)
        total_energy2 = np.array(ke2) + np.array(int_e2)
        energies_comp.append((np.average(total_energy[list(times).index(p_stop):]) - total_energy[0])/total_energy[0])
        energies2_comp.append((np.average(total_energy2[list(times).index(p_stop):]) - total_energy2[0])/total_energy2[0])
        energies.append((np.average(total_energy[list(times).index(p_stop):]) - total_energy[0]))
        energies2.append((np.average(total_energy2[list(times).index(p_stop):]) - total_energy2[0]))
    
    plt.scatter(kw, energies, color='blue')
    plt.scatter(kw, energies2, color='orange')
    plt.ylabel('Energy (nK)')
    plt.xlabel(r'$k_z$')
    plt.legend(['Total excess energy','Core excess energy'])
    plt.show()

    plt.scatter(kw, energies_comp, color='blue')
    plt.scatter(kw, energies2_comp, color='orange')
    plt.ylabel(r'$E_e/E_0$')
    plt.xlabel(r'$k_z$')
    plt.legend(['Total excess energy','Core excess energy'])
    plt.show()





    
        





