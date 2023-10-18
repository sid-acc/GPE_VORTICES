import matplotlib
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

#from mayavi import mlab

from typing import List
def avg(l: List[int]) -> float:

    return sum(l)/len(l)


import helpers
import kw_dr
import varicose

hbar = 1.05457 *10**(-34)
hbar2 = 1.054 *10**(-25)
hbar3 = 1.05457 *10**(-31)
hbar4 = 1.054 *10**(-22)
kb =  1.38*(10**(-32))
kb2 = 1.38*(10**(-26))
mass = 87 * 1.66 * 10**(-27)
a0 = 5.29 * 10**(-5)
r = 16

timestep = 5.0e-3
#plt.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
#matplotlib.rcParams['mathtext.fontsize'] = '12'
#plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
#plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams["font.family"] = "Times New Roman"
matplotlib.rcParams['font.size'] = 12

def mu_xi_def(xi):
    '''
    Definition of mu (h^2)/(2 m k_b xi^2)
    '''
    return (hbar2**2)/(2*mass*(xi**2)*kb2)

def plot_exp_wave_vel(n):
    a = np.array(range(0, 5000, 100))
    g = 4 * np.pi * a * a0 *(hbar**2)*(1.0/(39 * 1.66 * 10**(-27)))
    v = np.sqrt(g*n/(39 * 1.66 * 10**(-27)))
    plt.plot(a, v, color='orange')

    a_test = [500, 1000, 2000, 4000]
    v_test = [0.00113, 0.00105, 0.00129, 0.00186]
    print(v[11])
    plt.scatter(a_test, v_test)
    plt.show()
    

def dens(N, R, L):
    return N/(np.pi*(R**2) * L)


def exp_xi(R, N,L, a_s):
    vol = np.pi*(R**2)*L
    n = N/vol
    print(n)
    return 1/(2*np.sqrt(np.pi*a_s*(a0)*n))   

def gn(a_s, dens):
    return (4*dens*np.pi*a_s*a0*(hbar2**2)/(mass*kb2))




#def dens(R, L, N)


def first_moment(midline, axis):
    fm_loc = 0
    norm = 0
    for i in range(0, len(midline)):
        fm_loc += axis[i]*abs(midline[i])#*(axis[j + 1] - axis[j])
        norm += midline[i]#*(axis[j + 1] - axis[j])
    #norm = 1
    fm_loc /= norm
    return fm_loc


def gaussian_perturbation(NX, NY, NZ, Xrange):
    X = np.linspace(-(Xrange/2), Xrange/2, NX)
    sizes = np.full(NX, 10)
    peak = []
    times = [1, 2, 3, 4, 6, 8]
    for i in times:
        if i != 10: 
            dens_filename = f'OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-07-11T11-50-42\\I_dens.t_+{i}.000e+00.csv'
        else:
            dens_filename = f'OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-07-11T11-50-42\\I_dens.t_+1.000e+01.csv'
        slices = helpers.get_data(dens_filename, NX, NY, NZ)
        midline = slices[(int) (NZ/2), (int) (NY/2), :]
        peak_idx = midline[(int) (len(midline)/2):].argmax()
        peak_pos= X[(int) (len(midline)/2):][peak_idx]
        peak_pos2 = first_moment(midline, X)
        #if i >0 and peak_pos <= peak[(int) ((i-2)/2)]: 
            #peak_pos = X[(int) (len(midline)/2) + peak_idx + 5:][midline[(int) (len(midline)/2) + peak_idx + 5:].argmax()]
            #plt.scatter(X[peak_idx + 10:], midline[peak_idx + 10:])
            #plt.show()
        #plt.scatter(X, midline, sizes)
        #plt.show()
        peak.append(peak_pos)
        
    
    popt, pcov = curve_fit(helpers.f1, times, peak)
    print(peak, popt)
    plt.scatter(times, peak)
    plt.plot(times, helpers.f1(np.array(times), *popt), color='orange')
    plt.show()
    print((popt[0]/1000)/(2*(16 * 10**(-6))))

def shake_analysis(NX, NY, NZ, Xrange, Zrange):
    Z = np.linspace(-(Zrange/2), Zrange/2, NX)
    sizes = np.full(NZ, 10)
    times1 = np.array(['0.000', '1.500', '3.000', '4.500', '6.000', '7.500', '9.000', '1.050', '1.200', '1.350', '1.500', '1.650', '1.800', '1.950', '2.100','2.300', '2.500'])
    times2 = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 195, 210, 230, 250])
    coms = []
    data = ''
    for i, t in enumerate(times1):
        exp = '00'
        if i >=1 and i <=6:
            exp = '01'
        elif i > 6:
            exp = '02'
        dens_filename = f'OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-07-10T17-10-46\\I_dens.t_+{t}e+{exp}.csv'
        slices = helpers.get_data(dens_filename, NX, NY, NZ)
        midline = slices[:, (int) (NY/2), (int) (NX/2)]
        #plt.scatter(Z, midline, sizes)
        #plt.show()
        coms.append(first_moment(midline, Z))
        data += f'{{{times2[i]}, {coms[i]}}},'
        
        #coms.append(Z[midline.argmax()])
    #popt, pcov = curve_fit(f2, times2[2:], coms[2:])
    plt.scatter(times2, coms)
    print(data)
    #plt.plot(np.array(range(20, 400)), f2(np.array(range(20, 400)), *popt), color='orange')
    #print(coms)
    plt.show()

def kick_plot():
    N_list = []
    w_list = []

    N_list2 = [1.3e3, 1.3e4, 1.0e5]
    w_list2 = [3.36, 4.25, 9.32]
    errors = [0.09, 0.27, 0.15]
    with open('OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\Default Dataset.csv', 'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            N_list.append((float) (row[0]))
            w_list.append((float) (row[1]))
        #print(N_list, w_list)

    fit = np.polyfit(N_list, w_list, 3)
    p = np.poly1d(fit)
    x = np.linspace(1e3, 1.3e5, 500)

    plt.scatter(N_list, w_list, color='blue')
    #plt.plot(x, p(x), color='blue')
    plt.scatter(N_list2, w_list2, color='orange')
    plt.errorbar(N_list2, w_list2, yerr=errors, color='orange', ls='none')
    plt.xlabel('N')
    plt.ylabel('w')
    plt.xscale('log')
    plt.yscale('log')
    
    #plt.show()


        #es.append(pe + ke + int_e)
    
    #plt.plot(times2, es/es[0])
    #plt.show()


def kzxi(nodes, L, xi):
    return (nodes/L)*xi*np.pi


def linear_response(nodes, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, start, stop, step):
    '''
    Plots response of KW amplitude as function of potential amplitude
    '''
    directory = f'V:\\VortexSimulationData\\\kw_linresp2'
    pot_amps = []#np.linspace(0.2, 1.2, 6)
    amps = []
    err = []

    for sub_dir in os.scandir(directory):
        print(sub_dir.path)
        pot_amp = helpers.get_input_param(sub_dir.path, '-ShakeAmplitude')
        if pot_amp <= 0.4:
            times1, times2 = helpers.times_array(start, stop, step)
            x, y, z = kw_dr.vc_data(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times1, sub_dir.path)
            val = kw_dr.vc_sin_fit(x, y, z, radius, times2, nodes, height)#kw_dr.kw_spectrum(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, nodes, sub_dir.path, False, start, stop, step)#
            amps.append(val[0])
            err.append(val[1])
            pot_amps.append(pot_amp)
    
    #print(pot_amps, amps)
    popt, pcov = curve_fit(helpers.f1, pot_amps, np.array(amps))
    print(popt, np.sqrt(pcov[1][1]))
    plt.scatter(pot_amps, np.array(amps),color='blue')
    x = np.linspace(min(pot_amps), max(pot_amps), 50)
    plt.plot(x, helpers.f1(x, *popt), color='blue', linestyle = 'dashed')
    plt.errorbar(pot_amps, amps, yerr=err, ls='none', color='blue')
    plt.xlabel('Potential Strength (nK)')
    #plt.ylabel(r'$A/R$')
    plt.show()


def get_kwdr_data(filename, xi, mu):
    '''
    Gets wave number, box length, KW freq, and errors from file and normalizes units
    '''
    kz_list = []
    amps = []
    err = []
    with open(filename, 'r') as csvfile:
        csvr = csv.reader(csvfile)
        for row in csvr:
            if any(row):
                kz_list.append(kzxi((float)(row[0]),(float)(row[1]), xi))
                #amps.append((hbar2*(0.001)/(mu*kb2)) * ((float)(row[2])))
                amps.append((float)(row[2]))
                #err.append((hbar2*(0.001)/(mu*kb2)) * ((float)(row[3])))
                err.append((float)(row[3]))
    return kz_list, amps, err
     

def plot_mult_dr(directory, NX, NY, NZ, Xrange, Yrange, Zrange): 
    '''
    Plots KW dispersion relations
    '''
    initials_dir = directory + '\\Initials'
    data_dir = directory + '\\Widths'
    mu_list = []
    dens_list = []
    xi_list = []
    min_list = []
    max_list = []

    for sub_dir in os.scandir(initials_dir):
        a_s = helpers.get_input_param(sub_dir.path, '-ScatteringLength')
        N = helpers.get_input_param(sub_dir.path, '-AtomNumber')
        R = helpers.get_input_param(sub_dir.path, '-Radius')
        L = helpers.get_input_param(sub_dir.path, '-Length')
        n = dens(N, R, L)
        dens_list.append(n)
        xi = kw_dr.xi_fit(sub_dir.path, NX, NY, NZ, Xrange, True)
        print(xi)
        xi_list.append(xi)
        mu_list.append(mu_xi_def(xi))
        print(mu_xi_def(xi))

    print(dens_list, mu_list, np.array(xi_list))

    for i, file in enumerate(os.scandir(data_dir)):
        kz_list, amps, err = get_kwdr_data(file.path, xi_list[i], mu_list[i])
        plt.scatter(kz_list, np.array(amps), color = 'blue')
        plt.errorbar(kz_list, np.array(amps), color = 'blue',yerr=np.array(err), ls='none')
        min_list.append(min(kz_list))
        max_list.append(max(kz_list))

    kz_min = min(min_list)
    kz_max = max(max_list)
    x = np.linspace(kz_min, kz_max, 100)
    

    def exp_dr(arr, a, b, rad, ma):
        num = scipy.special.kv(2, a*rad*arr) + scipy.special.kv(0, a*rad*arr)
        den = scipy.special.iv(2, a*rad*arr) + scipy.special.iv(0, a*rad*arr)
        chi = num/den

        p1 = np.log(np.exp(1)/((2/np.pi)*np.arctan((np.pi/2)*b*arr*np.exp(1))))
        prefactor = (hbar2**2)*((arr/xi_list[0])**2)/(2*ma) *(1/(mu_list[0]*kb2*2*np.pi))
        return prefactor*(p1 - chi)


    st_kz =  np.array([0.3645, 0.1728,  0.4962, 0.6131, 0.554, 0.2982,  0.2327, 0.4262,0.1385])
    st_omega = np.array([62.6, 16.6, 106.4, 167, 136, 43.42, 27.6, 86.42, 11.66])
    st_xi = 3.5
    st_mu = mu_xi_def(3.5)
    st_kz = st_kz*st_xi
    st_omega = (hbar2*(0.001)/(st_mu*kb2))*st_omega
    y = np.linspace(min(st_kz), max(st_kz), 100)
    
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.plot(x, exp_dr(x, 1, 0.7095, 16, 87 * 1.66 * 10**(-27)), color='purple', ls='dashed')
    #plt.plot(x, (hbar2**2)*((x/xi_list[0])**2)/(4*np.pi*mass*mu_list[0]*kb2), color = 'darksalmon')
    x = x[:(int)(0.25*len(x))]
    #plt.plot(x, (hbar2**2)*((x/xi_list[0])**2)*np.log(1/(x))/(4*np.pi*mass*mu_list[0]*kb2), color = 'orange')
    #plt.plot(y, exp_dr(y, 1, 0.7095, 60, 12 * 1.66 * 10**(-27)), color='blue', ls='dashed')
    #plt.scatter(st_kz, st_omega)
    plt.xlabel(r'$k_z \xi$')
    plt.ylabel('Width (Hz)')
    #plt.ylabel(r'$\hbar\omega/\mu$')
    #plt.legend(['Simulation', 'Eq. 7', 'Single Particle in a Box', 'Eq. 4'])
    plt.show()


def plot_3d(dens_filename, NX, NY, NZ, Xrange, Yrange, Zrange):
    '''
    Gets 3D plot of wavefunction density, need to uncomment import mayavi in file header to use
    '''
    slices = helpers.get_data(dens_filename, NX, NY, NZ, False)
   
    fig = mlab.figure(size=(1000,1000))
    mlab.contour3d(slices, contours=3, opacity=0.1, extent=[-Zrange/2, Zrange/2, -Xrange/2, Xrange/2,  -Yrange/2, Yrange/2])
    mlab.axes()
    mlab.view(90, 180, distance=140)
    mlab.xlabel('Z')
    mlab.zlabel('Y')
    mlab.ylabel('X')
    mlab.show()
    #mlab.savefig('example4.png')

def plot_lengths(height, radius, n, NX, NY, NZ, Xrange, Yrange, Zrange, dir, start, stop, step):
    '''
    Plots KW amplitude over time (can plot actual length or amplitude or sin fit amplitude)
    '''
    times1, times2 = helpers.times_array(start, stop, step)
    #lengths = kw_dr.get_lengths(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, dir, start, stop, step)
    x, y, z = kw_dr.vc_data(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times1, dir)
    val = kw_dr.vc_sin_fit(x, y, z, radius, times2, n, height)
    for i, v in enumerate(val[2]):
        if v == 0:
            val[2][i] = (val[2][i-1] + val[2][i+1])/2
    plt.plot(times2, val[2], color='blue')
    plt.xlabel('Time (ms)')
    plt.ylabel('A/R')
    plt.show()    

def time_series_data(nodes, length, radius, NX, NY, NZ, Xrange, Yrange, Zrange, exp_mean, start, stop, step):
    '''
    Puts amplitudes for a frequency range for a particular KW mode in a file
    '''
    times1, times2 = helpers.times_array(start, stop, step)
    _, data_times = helpers.times_array(start, stop, step)
    freqs = []
    directory = f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\kw_{nodes}m_{length}l'
    data = np.empty([len(data_times), len(list(os.scandir(directory)))])
    errors = np.empty([len(data_times), len(list(os.scandir(directory)))])
    
    for i, sub_dir in enumerate(os.scandir(directory)):
        print(sub_dir.path)
        freqs.append(helpers.get_input_param(sub_dir.path, '-ShakeFrequency'))
        x, y, z = kw_dr.vc_data(length, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times1, sub_dir.path)
        amps = kw_dr.vc_sin_fit(x, y, z, radius, times2, nodes, length)
        for j, dt in enumerate(data_times):
            if dt != times2[len(times2)-1]:
                data[j][i] = (amps[2][list(times2).index(dt)] + amps[2][list(times2).index(dt)+1])/2
            else:
                data[j][i] = amps[2][list(times2).index(dt)]
            
    with open(f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\time_series_kzdep_13m.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(freqs)
        for row in data: 
            #print('yee')
            writer.writerow(row)  
   

    
def time_series_plot(filename, data_times, exp_mean, strength):
    '''
    Plots spectrum for particular KW mode at multiple times
    '''
    data_times_str = []
    for dt in data_times:
        data_times_str.append(f'{dt} ms')
    freqs = []
    data = []
    peak_pos = []
    width = []
    pp_err = []
    width_err = []
    with open(filename, 'r') as csvfile:
        csvr = csv.reader(csvfile)
        for i, row in enumerate(csvr):
            if any(row):
                for j, _ in enumerate(row):
                    row[j] = float(row[j])
                print(row)
                if i == 0:
                    freqs = row
                else:
                    data.append(row)
    freqs2 = np.linspace(min(freqs), max(freqs), 50)
    sizes = np.full(len(freqs), 20)
    #plt.scatter(freqs, max(amps))
    #fig, ax = plt.subplots()
    g_lines = []
    for i, dt in enumerate(data_times):
        ri = i #+ (int)((data_times[0] -30)/(data_times[1] - data_times[0]))
        print(dt)
        popt, pcov = kw_dr.gaussian_fit(exp_mean, 80, freqs, data[ri])
        peak_pos.append(popt[0])
        pp_err.append(np.sqrt(pcov[0][0]))
        width.append(helpers.gaussian_width(popt, pcov)[0])
        width_err.append(helpers.gaussian_width(popt, pcov)[1])
        if  dt == 60 or dt == 80 or dt == 100:
            #plt.scatter(freqs, data[ri], sizes)
            #plt.xlabel('w (Hz)')
            #plt.ylabel('A/R')
            #plt.legend(data_times_str)
            #plt.errorbar(freqs, data[i], errors[i])
            #g_lines.append(ax.plot(freqs2, helpers.gaussian(freqs2, *popt), label=data_times_str[i])[0])
            print(popt)
    
    #ax.legend(handles=g_lines)
    #plt.show()
    
    plt.plot(data_times, np.array(peak_pos))
    plt.xlabel('Time (ms)')
    plt.ylabel('Peak Amplitude')
    #plt.errorbar(data_times, peak_pos, yerr = pp_err, ls='none')
    #plt.show()

    #plt.plot(data_times, width)
    #plt.xlabel('Time (ms)')
    #plt.ylabel('Width (Hz)')
    #plt.errorbar(data_times, width, yerr = width_err, ls='none', color = 'blue')
    #plt.show()
                  
    return

def mult_time_series_plot(exp_mean):
    times1, times2 = helpers.times_array(40, 125, 5)
    strengths = [7, 13]
    
    for i, s in enumerate(strengths):
        filename = f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\time_series_kzdep_{s}m.csv'
        time_series_plot(filename, times2, exp_mean + i*15, 0.2)
    
    plt.legend([r'$k_z = 7\pi/L$', r'$k_z = 13\pi/L$'])
    plt.show()



def mult_freq_amp_plot(height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, n, exp_mean, start, stop, step):
    directories = ['C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\kw_7m_52l', 'V:\\VortexSimulationData\\kw_7m_52l']
    g_lines = []

    fig, ax = plt.subplots()
    for directory in directories:
        g_lines.append(kw_dr.freq_amp_plot(directory, height, radius, NX, NY, NZ, Xrange, Yrange, Zrange, n, exp_mean, start, stop, step, ax))
    
    ax.legend(handles=g_lines)
    plt.show()
    
    
def spectrum_linresp_data(strength, nodes, length, radius, NX, NY, NZ, Xrange, Yrange, Zrange, start, stop, step):
    '''
    Puts the response of a KW spectrum as a function of potential amplitude as a function of time
    '''
    times1, times2 = helpers.times_array(start, stop, step)
    freqs = []
    directory = f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\kw_{nodes}m_{length}l_{strength}'
    amps = np.empty([len(list(os.scandir(directory)))])
    errors = np.empty([len(list(os.scandir(directory)))])

    for i, sub_dir in enumerate(os.scandir(directory)):
        print(sub_dir.path)
        freqs.append(helpers.get_input_param(sub_dir.path, '-ShakeFrequency'))
        x, y, z = kw_dr.vc_data(length, radius, NX, NY, NZ, Xrange, Yrange, Zrange, times1, sub_dir.path)
        val = kw_dr.vc_sin_fit(x, y, z, radius, times2, nodes, length)
        amps[i] = (np.mean(val[2]))
        errors[i] = (np.std(val[2]))
    
    plt.scatter(freqs, amps)
    plt.show()

    with open(f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\spectrum_linresp.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow([strength])
        writer.writerow(freqs)
        writer.writerow(amps)  


def spectrum_linresp_plot(filename: str, exp_mean: float):
    '''
    Plots the response of a KW spectrum as a function of potential amplitude as a function of time
    '''
    strengths = []
    freqs = []
    amps = []
    peaks = []
    peak_err = []
    widths = []
    width_err = []
    magnitude = []

    with open(filename, 'r') as csvfile:
        csvr = csv.reader(csvfile)
        for i, row in enumerate(csvr):
            if any(row):
                row = np.array(list((filter(None, row))))
                #print(row)
                if len(list(row)) == 1:
                    strengths.append(float(row[0]))
                elif float(row[0]) >= 0.5: 
                    freqs.append(row.astype(np.float))
                else:
                    amps.append(row.astype(np.float))

    for i, freq in enumerate(freqs):
        popt, pcov = kw_dr.gaussian_fit(exp_mean, 80, freq, amps[i])
        peaks.append(popt[0])
        peak_err.append(np.sqrt(pcov[0][0]))
        widths.append(helpers.gaussian_width(popt, pcov)[0])
        width_err.append(helpers.gaussian_width(popt, pcov)[1])
        magnitude.append(max(amps[i]))
        plt.scatter(freq, amps[i], color='blue')
        freqs2 = np.linspace(min(freq), max(freq), 50)
        plt.plot(freqs2, helpers.gaussian(freqs2, *popt), color='blue', linestyle='dashed')
        plt.xlabel('Drive Frequency (Hz)')
        plt.ylabel('A/R')
        plt.title(strengths[i])
        plt.show()
    

    plt.scatter(strengths, peaks, color = 'blue')
    plt.errorbar(strengths, peaks, peak_err, ls='none', color='blue')
    plt.xlabel('Drive Strength (nK)')
    plt.ylabel('Peak Frequency (Hz)')
    plt.show()

    plt.scatter(strengths, widths, color = 'blue')
    plt.errorbar(strengths, widths, width_err, ls='none', color='blue')
    plt.xlabel('Drive Strength (nK)')
    plt.ylabel('Width (Hz)')
    plt.show()

    plt.scatter(strengths, magnitude, color = 'blue')
    #lt.errorbar(strengths, p, peak_err, ls='none', color='blue')
    plt.xlabel('Drive Strength (nK)')
    plt.ylabel('Peak Amplitude (A/R)')
    strengths.sort()
    magnitude.sort()
    str_order = list(strengths[:len(strengths)-2])
    mag_order = list(magnitude[:len(strengths)-2])
    print(str_order)
    popt,pcov = curve_fit(helpers.f1, str_order, mag_order)
    plt.plot(str_order, helpers.f1(np.array(str_order), *popt), color='blue', ls='dashed')
    plt.show()

def integrated_density_plot(directory, height, NZ, Zrange):
    times = [0, 60, 100]
    for t in times:
        print(t)
        st = '{:.3e}'.format(t)
        filename = directory + f'\\I_dens.t_+{st}.csv'
        slices = helpers.get_data(filename, NX, NY, NZ, False)
        z, mm = kw_dr.monopole_moment(slices, height, NZ, Zrange)
        plt.plot(z, mm)
    
    plt.legend(['0 ms', '60ms', '100ms'])
    plt.xlabel('z (um)')
    plt.show()








           
NX = 128
NY = 128
NZ = 128


h = '13'
m = '53'
s = '01'
dens_filename = f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-09-18T14-17-47\\I_dens.t_+1.000e+02.csv'
phase_filename = f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\kw_7m_100l\\gpeSolver_RK4-2023-08-30T18-47-00\\I_phase.t_+6.500e+01.csv'
pot_filename = f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-09-21T00-00-12\\I_td_pot.t_+1.000e+02.csv'

#filename = 'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\spectrum_linresp.csv'
#spectrum_linresp_plot(filename, 15)
#slices = helpers.get_data(pot_filename, NX, NY, NZ, True)
#slices2 = helpers.get_data_radius(slices, 40, 40, NX, NY, NZ, 5)
#helpers.plot_slices(slices, NX, NY, NZ,40, 40, 60)

#plot_3d(dens_filename, NX, NY, NZ, 40, 40, 60)
#integrated_density_plot('C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-09-18T14-17-47', 52, NZ, 60)
#vortex_core(slices, 26, 16, NX, NY, NZ, 40,40,40)
#xi_fit(slices, NX, NY, NZ, 40, True)
#print(exp_xi(20, 60, 2e4, 250))
#gaussian_perturbation(NX, NY, NZ, 40)
#plot_exp_wave_vel()
#shake_analysis(NX, NY, NZ, 40, 60)
#kick_plot()

#fig, ax = plt.subplots()
#directory = 'V:\\VortexSimulationData\\kw_dr_ld_toostrong\\kw_11m_52l'
#kw_dr.freq_amp_plot(directory, 52, 16, NX, NY, NZ, 40, 40, 60, 11, 20, 100, 125, 5, ax)

#dir = 'V:\VortexSimulationData\kw_dr_ld_toostrong\kw_11m_52l\gpeSolver_RK4-2023-09-14T22-24-13'
#times1, times2 = helpers.times_array(100, 125, 5)
#print(kw_dr.vc_data(52, 16, NX, NY, NZ, 40, 40, 60, times1, dir))

dir = 'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-09-28T18-03-31'
varicose.varicose_spectrum(dir, 52, 16, NX, NY, NZ, 40, 40, 60, 0, 125, 5)

#dir= 'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-09-22T20-49-00'
#val = kw_dr.kw_spectrum(52, 16, NX, NY, NZ, 40, 40, 60, 7, dir, False, 0, 1000, 25)
#spectrum_linresp_data(0.125, 7, 52, 16, NX, NY, NZ, 40, 40, 60, 60, 100, 5)
#dir = 'C:\\Users\\vishv\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-08-31T16-29-14'
#xi = kw_dr.xi_fit(dir, NX, NY, NZ, 105, True)
#print(mu_xi_def(xi))
#mult_freq_amp_plot(52, 16, NX, NY, NZ, 40, 40, 60, 7, 15, 60, 100, 5)
#plot_dr(NX, NY, NZ, 40, 40, 40, 100, 0.4782, 16)
#linear_response(7, 26, 16, NX, NY, NZ, 40, 40, 40, 0, 100, 5)
#dir = 'C:\\Users\\vishv\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-08-21T17-16-11'
#kw_dr.energy(dir, NX, NY, NZ, 40, 40, 40, 100, 0, 0, 100, 5)
#kw_dr.get_adb_freq(26, 16, NX, NY, NZ, 40, 40, 40, 9, 40, 100, 2.5)
#print(gn(100)/kb2)
#plot_mult_dr('C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\kw_dr2', NX, NY, NZ, 40, 40, 60)
#plot_mult_dr('C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\kw_dr3', NX, NY, NZ, 40, 40, 60)
#plt.legend(['0.4 nK for 60 ms', '0.2 nK for 100 ms'])
#plt.show()
#kw_dr.phonon_fft(7, 52, NX, NY, NZ, 40, 40, 60, 0, 1000, 25)
#dir = 'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\gpeSolver_RK4-2023-09-22T20-49-00'
#plot_lengths(52, 16, 7, NX, NY, NZ, 40, 40, 60, dir, 0, 1000, 25)


#times1, times2 = helpers.times_array(0, 60, 5)
#varicose.varicose_spectrum('V:\\VortexSimulationData\\vw\\gpeSolver_RK4-2023-08-17T16-46-45', 26, 16, NX, NY, NZ, 40, 40, 40, 0, 80, 5)
#kw_dr.gen_exp_data(7, 16, 26, 50, 0, 250,5)

#dir = 'V:\\VortexSimulationData\\kw_7m_26l_lt\\gpeSolver_RK4-2023-08-14T13-43-17'
#times1, times2 = helpers.times_array(0, 250, 5)
#kw_dr.vortex_core_animation(26, 16, NX, NY, NZ, 40, 40, 40, times1, dir)

#kw_dr.kw_energy_plot('V:\\VortexSimulationData\\kw_energy_linresp', NX, NY, NZ, 40, 40, 40, 100, 0, 100, 5, 40)
#time_series_data(13, 52, 16, NX, NY, NZ, 40, 40, 60,40, 40, 125, 5)
#filename = f'C:\\Users\\vishv\\OneDrive - California Institute of Technology\\Junior\\SURF\\gpeSolver_RK4\\x64\\Debug\\Outputs\\time_series_100msdrive02.csv'
#times1, times2 = helpers.times_array(40, 100, 5)
#time_series_plot(filename, times2, 5)

#mult_time_series_plot(15)




