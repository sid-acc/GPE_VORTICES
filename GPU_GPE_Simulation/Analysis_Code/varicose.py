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
import math

import helpers
import kw_dr

def core_size(dens, phase, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange):
    core = []
    X = np.linspace(-(Xrange/2), Xrange/2, NX)
    Y = np.linspace(-(Yrange/2), Yrange/2, NY)
    Z = np.linspace(-(Zrange/2), Zrange/2, NZ)
    center_x, center_y, z_array = kw_dr.vortex_core(phase, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange)
    for i, z in enumerate(z_array):
        dens_slice = dens[list(Z).index(z)]
        x_midline = dens_slice[list(Y).index(center_y[i])]
        y_midline = dens_slice[:, list(X).index(center_x[i])]
        d_midline1 = []
        d_midline2 = []
        for j in range(NX):
            d_midline1.append(dens_slice[j][j])
            d_midline2.append(dens_slice[NX-j-1][j])

        midline = (np.array(x_midline) + np.array(y_midline) + np.array(d_midline1) + np.array(d_midline2))/4
        with warnings.catch_warnings():
            while True:
                warnings.simplefilter("error", RuntimeError)
                try:
                    #popt, pcov = curve_fit(helpers.f, X[(int) (NX/2) + 12:NX-35], midline[12:(int) (NX/2 - 35)])
                    popt2, pcov2 = curve_fit(helpers.f, X[(int) (NX/2 - 18):(int) (NX/2)], midline[(int) (NX/2 - 18):(int) (NX/2)])
                    core.append(popt2[2]*np.sqrt(2))
                    break
                except RuntimeError:
                    print('here')
                    core.append(core[i-1])
                    break

        #plt.scatter(X, midline)
        #plt.plot(X[(int) (NX/2) + 12:NX-35] - Xrange/2, helpers.f(X[(int) (NX/2) + 12:NX-35], *popt), color='orange')
        #plt.plot(X[(int) (NX/2 - 18):(int) (NX/2)], helpers.f(X[(int) (NX/2 - 18):(int) (NX/2)], *popt2), color='green')
        #plt.show()
        #xi = (popt[2] + popt2[2])*(np.sqrt(2))/2
        
        #if math.isinf(pcov[0][0]) or np.average(pcov) > xi*0.1:
         #   xi = popt2[2]*np.sqrt(2)
        #if math.isinf(pcov2[0][0])or np.average(pcov2) > xi*0.1:
        #    xi = popt[2]*np.sqrt(2)
        #print(xi)
        #core.append(popt2[2]*np.sqrt(2))
    #plt.plot(z_array, core)
    #plt.show()
    return z_array, core

    
def core_size_data(directory, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times):
    cores = []
    for t in times:
        phase_filename = f'{directory}\\I_phase.t_+{t}.csv'
        phase = helpers.get_data(phase_filename, NX, NY, NZ, False)
        dens_filename = f'{directory}\\I_dens.t_+{t}.csv'
        dens = helpers.get_data(dens_filename, NX, NY, NZ, False)
        z_array, core_t = core_size(dens, phase, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange)
        #z_array, core_t = kw_dr.monopole_moment(dens, height, NZ, Zrange)
        cores.append(core_t)
        plt.plot(np.array(core_t)-np.array(cores[0]))
        plt.show()
    cores = np.array(cores) - np.array(cores[0]) 
    return z_array, cores

def varicose_spectrum(dir, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, start, stop, step):
    times1, times2 = helpers.times_array(start, stop, step)
    z_array, data = core_size_data(dir, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times1)
    kz, omega, ft2d, ft1d = helpers.general_fft(times2, z_array, data)
    helpers.plot_fft(kz, times2, ft1d)
    helpers.plot_fft(kz, omega, ft2d)

def vfreq_sweep(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, nodes, exp_mean, start, stop, step):
    directory = f'V:\\VortexSimulationData\\vw_{nodes}m_{height}l'
    freqs = []
    amps = []
    stds = []

    for sub_dir in os.scandir(directory):
        print(sub_dir.path)

    return 





