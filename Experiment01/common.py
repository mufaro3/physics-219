import numpy as np
import array
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

def load_raw_data(filename):
    data = np.loadtxt(filename,
                      delimiter=',',
                      comments='#',
                      usecols=(3,4),
                      skiprows=1)
    xvalues = data[:,0]
    yvalues = data[:,1]
    indices = np.arange(len(xvalues))
    return (xvalues,yvalues,indices)

def plot_raw_data(raw_data, title):
    t, y, indices = raw_data
    
    plt.figure()
    plt.scatter(t, y, marker='.')
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (V)')
    plt.title(title)
    plt.show()
    
    plt.figure()
    plt.scatter(indices, y, marker='.')
    plt.xlabel('Index')
    plt.ylabel('Voltage (V)')
    plt.title(title + ' - Voltage vs. Index')
    plt.show()

def isolate_noise_and_plot(raw_data, indices_range, y_range):
    x, y, indices = raw_data
    
    plt.figure()
    plt.xlim(indices_range[0], indices_range[1])
    plt.ylim(y_range[0], y_range[1])
    plt.scatter(indices, y, marker='.')
    plt.show()
    
    # 
    y_ave = np.mean(y[indices_range[0]:indices_range[1]])
    y_std = np.std(y[indices_range[0]:indices_range[1]])
    
    print('mean = ',y_ave,y_std)
    print('standard deviation = ', y_std)
    
    hist, bins = np.histogram(y[indices_range[0]:indices_range[1]],bins=20)
    plt.bar(bins[:-1], hist, width = bins[1]-bins[0])
    plt.ylim(0, 1.2 * np.max(hist))
    plt.xlabel('y_raw (Volts)')
    plt.ylabel('Number of occurences')
    plt.show()
    
    return y_std

def pack(A, p):
  # A is an array, and p is the packing factor
  B = np.zeros(len(A)//p)
  i = 1
  while i - 1 < len(B):
    B[i-1] = np.mean(A[p*(i-1):p*i])
    i += 1
  return B

def pack_data(data, p, noise):
    x_raw, y_raw, indices = data
    x = pack(x_raw, p)
    y = pack(y_raw, p)
    
    length = len(x)
    index  = np.arange(length)
    sigmax = np.zeros(length)
    
    sigmayraw  = noise
    sigmaymean = sigmayraw / np.sqrt(p)
    sigmay     = np.array([sigmaymean] * length)
    
    return (x, sigmax, y, sigmay)

def plot_packed_data(packed_data):
    x, dx, y, dy = packed_data
    
    plt.errorbar(x, y,yerr=dy, marker='.', linestyle='')
    plt.xlabel('Time (sec)')
    plt.ylabel('Voltage (V)')
    plt.title('Packed Data')
    plt.show()
    
def save_packed_data(packed_data, output_filename):
    x, dx, y, dy = packed_data
    
    header = [np.array(['Time','u[time]','Voltage','u[Voltage]']), 
              np.array(['(sec)','(sec)','(V)','(V)'])]
    df = pd.DataFrame(np.array([x , dx, y , dy]).transpose(), columns=header)   
    
    csv_data = df.to_csv(output_name, index = False)
    print('Packed Data Stored in ', output_filename)
    
def guess_fit(packed_data, amplitude_guess, frequency_guess, phase_guess, offset_guess=None):
    def sine_func(x, amplitude, freq, phase):
        return amplitude * np.sin(2.0 * np.pi * freq * x + phase)

    def offset_sine_func(x, amplitude, freq, phase, offset):
        return (amplitude * np.sin(2.0 * np.pi * freq * x + phase)) + offset
    
    # Names and units of data columns from fname
    x_name = "Time"
    x_units = "s"
    y_name = "Voltage"
    y_units = "V"

    # Modify to change the fitting function, parameter names and to set initial parameter guesses
    fit_function = sine_func
    
    param_names = ("amplitude", "frequency", "phase")
    guesses = (amplitude_guess, frequency_guess, phase_guess)
    
    if offset_guess is not None:
        fit_function = offset_sine_func
        param_names = ("amplitude", "frequency", "phase", "offset")
        guesses = (amplitude_guess, frequency_guess, phase_guess, offset_guess)
    
    x, dx, y, dy = packed_data
    
    # Define 500 points spanning the range of the x-data; for plotting smooth curves
    xtheory = np.linspace(min(x), max(x), 500)

    # Compare the guessed curve to the data for visual reference
    y_guess = fit_function(xtheory, *guesses)
    plt.errorbar(x, y, yerr=dy, marker=".", linestyle="", label="Measured data")
    plt.plot(
        xtheory,
        y_guess,
        marker="",
        linestyle="-",
        linewidth=1,
        color="g",
        label="Initial parameter guesses",
    )
    plt.xlabel(f"{x_name} [{x_units}]")
    plt.ylabel(f"{y_name} [{y_units}]")
    plt.title(r"Comparison between the data and the intial parameter guesses")
    plt.legend(loc="best", numpoints=1)
    plt.show()
    
    # calculate the value of the model at each of the x-values of the data set
    y_fit = fit_function(x, *guesses)

    # Residuals are the difference between the data and theory
    residual = y - y_fit

    # Plot the residuals
    plt.errorbar(x, residual, yerr=dy, marker=".", linestyle="", label="residuals")
    plt.xlabel(f"{x_name} [{x_units}]")
    plt.ylabel(f"Residual y-y_fit [{y_units}]")
    plt.title("Residuals using initial parameter guesses")
    plt.show()
    
    return x, y, dy, guesses, fit_function, param_names, x_name, y_name, x_units, y_units

def auto_fit(packed_data, fit_params):
    x, y, dy, guesses, fit_function, param_names, x_name, y_name, x_units, y_units = fit_params
    
    fit_params, fit_cov = curve_fit(
        fit_function, x, y, sigma=dy, 
        p0=guesses,absolute_sigma=True, maxfev=10**5)

    # Define the function that calculates chi-squared
    def chi_square(fit_parameters, x, y, sigma):
        dof = len(x) - len(fit_params)
        return np.sum((y - fit_function(x, *fit_parameters)) ** 2 / sigma ** 2) / dof

    # Calculate and print reduced chi-squared
    chi2 = chi_square(fit_params, x, y, dy)
    print("Chi-squared = ", chi2)

    # Calculate the uncertainties in the fit parameters
    fit_params_error = np.sqrt(np.diag(fit_cov))

    # Print the fit parameters with uncertianties
    print("\nFit parameters:")
    for i in range(len(fit_params)):
        print(f"   {param_names[i]} = {fit_params[i]:.3e} ± {fit_params_error[i]:.3e}")
    print("\n")

    # residual is the difference between the data and model
    x_fitfunc = np.linspace(min(x), max(x), len(x))
    y_fitfunc = fit_function(x_fitfunc, *fit_params)
    y_fit = fit_function(x, *fit_params)
    residual = y-y_fit

    # The size of the canvas
    fig = plt.figure(figsize=(7,15))

    # The scatter plot
    ax1 = fig.add_subplot(311)
    ax1.errorbar(x,y,yerr=dy,marker='.',linestyle='',label="Measured data")
    ax1.plot(x_fitfunc, y_fitfunc, marker="", linestyle="-", linewidth=2,color="r", label="Fit")
    ax1.set_xlabel(f"{x_name} [{x_units}]")
    ax1.set_ylabel(f"{y_name} [{y_units}]")
    ax1.set_title('Best Fit of Function to Data')

    # Show the legend. loc='best' places it where the date are least obstructed
    ax1.legend(loc='best',numpoints=1)

    # The residuals plot
    ax2 = fig.add_subplot(312)
    ax2.errorbar(x, residual, yerr=dy,marker='.', linestyle='', label="Residual (y-y_fit)")
    ax2.hlines(0,np.min(x),np.max(x),lw=2,alpha=0.8)
    ax2.set_xlabel(f"{x_name} [{x_units}]")
    ax2.set_ylabel(f"y-y_fit [{y_units}]")
    ax2.set_title('Residuals for the Best Fit')
    ax2.legend(loc='best',numpoints=1)

    # Histogram of the residuals
    ax3 = fig.add_subplot(313)
    hist,bins = np.histogram(residual,bins=30)
    ax3.bar(bins[:-1],hist,width=bins[1]-bins[0])
    ax3.set_ylim(0,1.2*np.max(hist))
    ax3.set_xlabel(f"y-y_fit [{y_units}]")
    ax3.set_ylabel('Number of occurences')
    ax3.set_title('Histogram of the Residuals')

    # Show the plot
    plt.show()
