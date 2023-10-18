import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.fft import fft, fft2, fftfreq, fftshift

hbar = 1.05457 *10**(-34)
hbar2 = 1.054 *10**(-25)
hbar3 = 1.05457 *10**(-31)
hbar4 = 1.054 *10**(-22)
kb =  1.38*(10**(-32))
kb2 = 1.38*(10**(-26))
mass = 87 * 1.66 * 10**(-27)
a0 = 5.29 * 10**(-5)
timestep = 5.0e-3

plt.rcParams['mathtext.rm'] = 'stix'
#plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
#plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
plt.rcParams["font.family"] = "STIXGeneral"
plt.rcParams['font.size'] = 12

def f(x, n1, a, xi):
    return n1*(np.tanh((x - a)/(np.sqrt(2)*xi)))**2 

def prop(x, a):
    return x * a

def f1(x, a, b):
    return a*x + b

def f2(x, z0, A, w, phi, b):
    return z0 + A*np.exp(-b*x)*np.sin(w*x + phi)

def gaussian(x, mu, b, a, c):
    return a*np.exp(-b*((x - mu)**2)) + c

def lorentzian(x, a, b, c):
    return a*(1/((x-b)**2 + c))

def sin_func(x, A, phi, w, c):
    return A*np.sin(w*x + phi) + c

def get_data(filename, NX, NY, NZ, e):
    '''
    Gets data from csv file and puts it in 3D array, if the file is energy, setting 'e' to True
    will normalize to correct units
    '''
    slices = np.empty((NZ, NY, NX))

    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter = ',')
        for i, row in enumerate(plots):
            row.remove(row[0])
            slices[(int)(i/NZ), i%NY] = row
    
    if e:
        slices = slices *(hbar3/(timestep*kb))
    
    return slices

def get_data_radius(slices, Xrange, Yrange, NX, NY, NZ, radius):
    '''
    Gets data within certain radius and sets the rest to zero
    '''
    
    X = np.linspace(-Xrange/2, Xrange/2, NX)
    Y = np.linspace(-Yrange/2, Yrange/2, NY)
    slices2 = list(np.zeros((NZ, NY, NX)))

    for i, z_slice in enumerate(slices):
        for j, y_slice in enumerate(z_slice):
            for k, _ in enumerate(y_slice):
                if X[k]**2 + Y[j]**2 <= radius**2:
                   slices2[i][j][k] = slices[i][j][k]
            
    return np.array(slices2)

def get_input_param(folder, param):
    '''
    Gets an input parameter from a data folder
    '''
    filename = folder + '\\Log.txt'
    with open(filename) as f:
        for line in f:
            line = line.strip()
            #print(line)
            if line[:(len(param))] == param:
                #print((float) (line[17:]))
                return (float) (line[len(param) + 2:])
            
def times_array(start, stop, step):
    '''
    Gives an int array and str array for times between start and stop 
    '''
    times1 = []
    times2 = np.linspace(start, stop, (int)(((stop-start)/step) + 1))
    for time in times2:
        times1.append('{:.3e}'.format(time))
    #print(times1)
    return times1, times2

def lorentzian_fit(x, y):
   
    popt, pcov = curve_fit(lorentzian, x, y)
    #plt.scatter(x, y, color = 'blue')
    #plt.plot(np.linspace(min(x), max(x), 100), lorentzian(np.linspace(min(x), max(x), 100), *popt), color = 'blue', linestyle='dashed')
    #plt.show()
    return popt, pcov


def gaussian_width(popt, pcov):
    '''
    Gets width of Gaussian from fit
    '''
    b = popt[1]
    db = np.sqrt(pcov[1][1])
    width = np.sqrt(1/(2*b))
    width_err = 0.5*((1/(2*b))*np.sqrt((1/(2*b))))*db
    return width, width_err

def plot_fft(h_axis, v_axis, data):
    '''
    Plots 1D and 2D ffts based on h_axis, v_axis input
    '''
    plt.pcolormesh(h_axis, v_axis, data/np.max(data))
    plt.colorbar()
    plt.xlabel('Wave index')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    
def fft_1d(w):
    '''
    Does 1D fft of 2D array w
    '''
    fft1d = []
    for s in w:
        wf_1d = fft(s)
        phase = np.cos(np.arctan(wf_1d.imag/wf_1d.real))
        fft1d.append(fftshift(abs(wf_1d)))
    return np.array(fft1d)

def fft_1d2(w):
    '''
    Does 2D fft of 2D array w
    '''
    fft1d = []
    #print(np.array(w[0]).size)
    for i in range(w[0].size):
        wf_1d = fft(w[:,i])
        #print(wf_1d.size)
        phase = np.cos(np.arctan(wf_1d.imag/wf_1d.real))
        fft1d.append(fftshift(phase))
    #print(fft1d.size)
    return np.array(fft1d)

def general_fft(times, z, data):
    '''
    Performs 1D and 2D ffts for 1D arrays times and z, 2D array data
    '''
    omega = fftfreq(len(times), 0.001*(times[1] - times[0]))
    omega = fftshift(omega)

    kz = fftfreq(len(z), ((z[len(z) - 1] - z[0])/len(z)))
    kz = fftshift(kz)*(2*(z[len(z) - 1] - z[0]))


    
    wf2d = fftshift(abs(fft2(data)))/(len(times)*len(z))
    wf1d = fft_1d(data)/len(z)

    return (kz, omega, wf2d, wf1d)

def plot_slices(slices, NX, NY, NZ, Xrange, Yrange, Zrange):
    '''
    Plots slices obtained from get_data function
    '''
    #xi = exp_xi(R, L, N, a_s)
    #slices = slices/((Xrange/NX) * (Yrange/NY)*(Zrange/NZ))
    #fig, ax = plt.subplots()
    X = np.linspace(-(Xrange/2), Xrange/2, NX)
    Y = np.linspace(-(Yrange/2), Yrange/2, NY)
    Z = np.linspace(-(Zrange/2), Zrange/2, NY)
    

    z_int = np.zeros([NX, NY])
    for z in slices:
        z_int += z
    
    y_int = np.zeros([NX, NZ])
    for i in range(NY):
        y_int += slices[:, i]
    #plt.pcolormesh(X, Z, slices[:, :,(int) (NZ/2)])
    plot = plt.pcolormesh(X, Z, slices[:,(int) (NZ/2)],cmap='magma')
    cbar = plt.colorbar(plot)
    cbar.ax.set_ylabel(r'Amplitude (nK)', rotation=270, labelpad=15)
    #plt.pcolormesh(X, Y, z_int)
    #plt.pcolormesh(X, Z, y_int)
    #plt.xlabel('x (um)')
    plt.xlabel('y (um)')
    plt.ylabel('z (um)')
    #plt.colorbar()
    plt.show()

    plt.pcolormesh(X, Z, slices[:,:,(int) (NZ/2)],cmap='magma')
    plt.xlabel('x (um)')
    plt.ylabel('z (um)')
    plt.show()

def plot_energy(times, ke_list, int_e_list):
    plt.plot(times, ke_list)
    #plt.plot(times2, pe_list)
    plt.plot(times, int_e_list)   
    plt.plot(times, np.array(ke_list) + np.array(int_e_list))
    plt.legend(['Kinetic', 'Interaction', 'Kinetic + Interaction'])
    plt.yscale('log')
    plt.xlabel('Time (ms)')
    plt.ylabel('Energy (nK)')
    plt.show() 

