# +
import numpy as np
import array
from matplotlib import pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from dataclasses import dataclass
from enum import Enum
from PIL import Image
import io
from abc import ABC, abstractmethod
from tabulate import tabulate

from IPython.display import display, Markdown, Latex


# -

@dataclass
class GraphingOptions:
    x_label: str = ''
    y_label: str = ''
    x_units: str = ''
    y_units: str = ''
    
    data_marker:      str   = '.'
    data_marker_size: int   = 2
    data_linestyle:   str   = ''
    data_alpha:       float = 0.80
    data_color:       str   = 'C0'
    
    model_marker:     str   = ''
    model_linestyle:  str   = '-'
    model_linewidth:  int   = 2
    model_alpha:      float = 1.0
    model_color:      str   = 'darkred' 
    
    data_round: int = 1

    def set_labels(self, xlabel=None, ylabel=None):
        if xlabel is None:
            plt.xlabel(f"{self.x_label} ({self.x_units})")
        else:
            plt.xlabel(xlabel)

        if ylabel is None:
            plt.ylabel(f"{self.y_label} ({self.y_units})")
        else:
            plt.ylabel(ylabel)
            
    def plot_data(self, x, y, y_uncert, label=None, color=None):
        plt.errorbar(x, y, yerr=y_uncert, 
                     marker     = self.data_marker,
                     markersize = self.data_marker_size,
                     linestyle  = self.data_linestyle,
                     alpha      = self.data_alpha,
                     color      = color if color is not None else self.data_color,
                     label      = label)
    
    def plot_model(self, model_x, model_y, model):
        plt.plot(model_x, model_y, 
                 marker    = self.model_marker, 
                 linestyle = self.model_linestyle, 
                 linewidth = self.model_linewidth,
                 alpha     = self.model_alpha,
                 color     = self.model_color, 
#                 label     = f'Fit - {model.as_equation_string()}')
                 label     = f'Fit')
    
    def plot_residuals(self, x, residuals, y_uncert):
        plt.errorbar(x, residuals, yerr=y_uncert, 
                     marker     = self.data_marker,
                     markersize = self.data_marker_size,
                     linestyle  = self.data_linestyle,
                     alpha      = self.data_alpha,
                     color      = self.data_color,
                     label      = "Residuals")
        plt.axhline(y=0, 
                    marker    = self.model_marker, 
                    linestyle = self.model_linestyle, 
                    linewidth = self.model_linewidth,
                    alpha     = self.model_alpha,
                    color     = self.model_color, 
                    label     = f'Fit')

    @staticmethod
    def save_graph_and_close():
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph = Image.open(buf)
        plt.close()
        return graph
        
    def default_title(self):
        return f'{self.y_label} vs. {self.x_label}, Round {self.data_round}'

@dataclass
class FitParameters(ABC):
    @abstractmethod
    def values(self): ...
    
    @abstractmethod
    def uncertainties(self): ...
    
    @abstractmethod
    def as_equation_string(self): ...
    
    @classmethod
    @abstractmethod
    def labels(): ...
    
    @classmethod
    @abstractmethod
    def fit_function(x, *args, **kwargs): ...
    
    def tabulate(self, units=None):
        def apply_units(label, i):
            if units is None or len(units) <= i:
                return label
            else:
                return f'{label} ({units[i]})'
        
        header = [ 'Measurement', 'Value', 'Uncertainty' ]
        rows = [ 
            [ apply_units(label, i), '%.3e' % value, '%.3e' % uncert ] \
                for i, (label, value, uncert) in \
                    enumerate(zip(self.labels(), 
                                  self.values(), 
                                  self.uncertainties())) 
        ]
        
        return tabulate(rows, header, tablefmt='grid')

# **Exponential Fit**
#
# $$f(x, A, \tau, C) = A \exp(-x/\tau) + C$$

@dataclass
class ExponentialFitParameters(FitParameters):
    amplitude:     np.float64 = 0
    time_constant: np.float64 = 0
    offset:        np.float64 = 0

    amplitude_uncert:     np.float64 = -1
    time_constant_uncert: np.float64 = -1
    offset_uncert:        np.float64 = -1

    @staticmethod
    def labels():
        return ('amplitude', 'time constant', 'offset')
    
    def values(self):
        return (self.amplitude, self.time_constant, self.offset)
    
    def uncertainties(self):
        return (self.amplitude_uncert, 
                self.time_constant_uncert, 
                self.offset_uncert)
    
    def as_equation_string(self):
        return '%.3e * exp(-x/%.3e) + %.3e' % self.values()
    
    @staticmethod
    def fit_function(x, amplitude, time_constant, offset):
        return amplitude * np.exp(-x/time_constant) + offset

# **Sinusoidal Fit**
#
# $$f(x,A,f,\phi) = A \sin (2 \pi f x + \phi)$$

@dataclass
class SinusoidalFitParameters(FitParameters):
    amplitude: np.float64 = 0
    frequency: np.float64 = 0
    phase:     np.float64 = 0

    # uncertainties (for later)
    amplitude_uncert: np.float64 = -1
    frequency_uncert: np.float64 = -1
    phase_uncert:     np.float64 = -1

    @staticmethod
    def labels():
        return ('amplitude', 'frequency', 'phase')
    
    def values(self):
        return (self.amplitude, self.frequency, self.phase)
    
    def uncertainties(self):
        return (self.amplitude_uncert,
               self.frequency_uncert,
               self.phase_uncert)
    
    def as_equation_string(self):
        return '%.3e * sin(2 * pi * %.3e + %.3e)' % self.values()
    
    @staticmethod
    def fit_function(x, amplitude, freq, phase):
        return amplitude * np.sin(2.0 * np.pi * freq * x + phase)

# **Offset Sinusoidal Fit**
#
# $$f(x,A,f,\phi,C) = A \sin (2 \pi f x + \phi) + C$$

@dataclass
class OffsetSinusoidalFitParameters(SinusoidalFitParameters):
    offset: np.float64 = 0

    offset_uncert: np.float64 = -1

    @staticmethod
    def labels():
        return ('amplitude', 'frequency', 'phase', 'offset')

    def values(self):
        return (super().amplitude, super().frequency, super().phase, self.offset)
    
    def uncertainties(self):
        return (super().amplitude_uncert,
                super().frequency_uncert, 
                super().phase_uncert, 
                self.offset_uncert)
    
    def as_equation_string(self):
        return super().as_equation_string() + ' + %.3e' % self.offset
    
    @staticmethod
    def fit_function(x, amplitude, freq, phase, offset):
        return (amplitude * np.sin(2.0 * np.pi * freq * x + phase)) + offset

# **Ringdown Fit**
#
# $$f(x, A, \tau, f, \phi) = A \exp(-x/\tau) \cos(2 \pi f x + \phi)$$

@dataclass
class RingdownFitParameters(FitParameters):
    amplitude:          np.float64 = 0
    time_constant:      np.float64 = 0
    resonant_frequency: np.float64 = 0
    phase:              np.float64 = 0
    
    # uncertainties
    amplitude_uncert:          np.float64 = 0
    time_constant_uncert:      np.float64 = 0
    resonant_frequency_uncert: np.float64 = 0
    phase_uncert:              np.float64 = 0
    
    @staticmethod
    def labels():
        return ('amplitude', 'time_constant', 'resonant_frequency', 'phase')
    
    def values(self):
        return (self.amplitude, 
                self.time_constant, 
                self.resonant_frequency, 
                self.phase)
    
    def uncertainties(self):
        return (self.amplitude_uncert,
                self.time_constant_uncert,
                self.resonant_frequency_uncert,
                self.phase_uncert)
        
    def as_equation_string(self):
        return '%.3e exp(-x/%.3e)cos(2 * pi * %.3e x + %.3e)' % self.values()
        
    @staticmethod
    def fit_function(x, amplitude, time_constant, resonant_frequency, phase):
        return amplitude * np.exp(-x/time_constant) * \
            np.cos(2.0 * np.pi * resonant_frequency * x + phase)

@dataclass
class RCResponseFitParameters(FitParameters):
    amplitude:     np.float64 = 0
    time_constant: np.float64 = 0
    offset:        np.float64 = 0
    
    # uncertainties
    amplitude_uncert:     np.float64 = 0
    time_constant_uncert: np.float64 = 0
    offset_uncert:        np.float64 = 0
    
    @staticmethod
    def labels():
        return ('amplitude', 'time_constant', 'offset')
    
    def values(self):
        return (self.amplitude, 
                self.time_constant,
                self.offset)
    
    def uncertainties(self):
        return (self.amplitude_uncert,
                self.time_constant_uncert,
                self.offset_uncert)
        
    def as_equation_string(self):
        return '%.3e / sqrt(1 + (2 * pi * x * %.3e)^2) + %.3e' % self.values()
        
    @staticmethod
    def fit_function(x, amplitude, time_constant, offset):
        return amplitude / np.sqrt(1 + (2 * np.pi * x * time_constant) ** 2) + offset

@dataclass
class LRCResponseFitParameters(FitParameters):
    amplitude:       np.float64 = 0
    gamma:           np.float64 = 0
    resonant_frequency: np.float64 = 0
    
    # uncertainties
    amplitude_uncert:       np.float64 = 0
    gamma_uncert:   np.float64 = 0
    resonant_frequency_uncert: np.float64 = 0
    
    @staticmethod
    def labels():
        return ('amplitude', 'gamma', 'resonant frequency')
    
    def values(self):
        return (self.amplitude, 
                self.gamma,
                self.resonant_frequency)
    
    def uncertainties(self):
        return (self.amplitude_uncert,
                self.gamma_uncert,
                self.resonant_frequency_uncert)
        
    def as_equation_string(self):
        return 'NONE'
        
    @staticmethod
    def fit_function(x, amplitude, gamma, resonant_frequency):
        return amplitude / np.sqrt(1 + (2 * np.pi / (gamma * x)) ** 2 *(x**2-resonant_frequency ** 2) ** 2)

# **2-Parameter Linear Fit**
#
# $$f(x, m, b) = mx + b$$

@dataclass
class LinearFitParameters(FitParameters):
    slope:     np.float64 = 0
    intercept: np.float64 = 0
    
    slope_uncert:     np.float64 = -1
    intercept_uncert: np.float64 = -1
    
    @staticmethod
    def labels():
        return ('slope', 'intercept')
    
    def values(self):
        return (self.slope, self.intercept)
    
    def uncertainties(self):
        return (self.slope_uncert, self.intercept_uncert)
    
    def as_equation_string(self):
        return '%.3e*x + %.3e' % self.values()
    
    @staticmethod
    def fit_function(x, slope, intercept):
        return slope * x + intercept


# **1-Parameter Linear Fit**
# $$f(x, m) = mx$$

@dataclass
class LinearFitParametersOne(FitParameters):
    slope:     np.float64 = 0
    slope_uncert:     np.float64 = -1
    
    @staticmethod
    def labels():
        return ('slope')
    
    def values(self):
        return (self.slope,)
    
    def uncertainties(self):
        return (self.slope_uncert,)
    
    def as_equation_string(self):
        return '%.3e*x' % self.values()
    
    @staticmethod
    def fit_function(x, slope):
        return slope * x


@dataclass
class FitModelResult:
    initial_guess_graph:           Image = None
    initial_guess_residuals_graph: Image = None
    autofit_graph:                 Image = None
    autofit_residuals_graph:       Image = None

    parameters:        FitParameters = None
    chi2:              np.float64    = None
    covariance_matrix: np.array      = None
    
    def tabulate(self, print_cov=False, units=None):
        print(self.parameters.tabulate(units=units))
        print('Chi^2 = %.3f' % self.chi2)
        
        if print_cov:
            print("Covariance Values:")
            for i, fit_covariance in enumerate(self.covariance_matrix):
                for j in range(i+1, len(fit_covariance)):
                    print(f"{self.parameters.labels()[i]} and {self.parameters.labels()[j]}: {self.covariance_matrix[i,j]:.3e}")
            print("\n")


VOLTAGE_VERSUS_TIME_GRAPH_OPTIONS = GraphingOptions(
    x_label='Time',
    y_label='Voltage',
    x_units='s',
    y_units='V'
)

def load_raw_data(filename, trim_range=None, plot=False, graphing_options=None):
    data = np.loadtxt(filename,
                      delimiter=',',
                      comments='#',
                      usecols=(3,4),
                      skiprows=1)
    xvalues = data[:,0]
    yvalues = data[:,1]
    indices = np.arange(len(xvalues))
    
    if trim_range is not None:
        xvalues_trimmed = xvalues[trim_range[0]:trim_range[1]]
        yvalues_trimmed = yvalues[trim_range[0]:trim_range[1]]
        indices_trimmed = indices[trim_range[0]:trim_range[1]]

    if plot:
        plt.figure()
        plt.scatter(xvalues, yvalues, marker='.')
        graphing_options.set_labels()
        plt.title('Raw Data: ' + graphing_options.default_title())
        plt.show()
        
        plt.figure()
        plt.scatter(indices, yvalues, marker='.')
        graphing_options.set_labels(xlabel='Index')
        plt.title(f'Raw Data: {graphing_options.y_label} vs. Index, Round {graphing_options.data_round}')
        if trim_range is not None:
            mask = (indices > trim_range[0]) & (indices < trim_range[1])
            plt.fill_between(indices, min(yvalues), max(yvalues), 
                             where = mask, 
                             color='green', alpha=0.1, label='Trimmed Range')
            plt.legend()
        plt.show()

    return (xvalues,yvalues),(xvalues_trimmed,yvalues_trimmed)

def calculate_uncertainty(raw_data, method="default",
                          manual_uncert=None,
                          indices_range=None, y_range=None, 
                          plot=False, graphing_options=None):
    if indices_range is None:
        indices_range = (0, len(raw_data[0]))
    
    x_trimmed, y_trimmed = \
        map(lambda a: a[indices_range[0]:indices_range[1]],
            (raw_data[0], raw_data[1]))
    indices_trimmed = np.arange(0, len(x_trimmed))
    
    data_trimmed = (x_trimmed, y_trimmed, indices_trimmed)
    
    x, y = raw_data
    indices = np.arange(0, len(x))
    
    if plot:  
        plt.figure()
        plt.xlim(indices_range[0], indices_range[1])
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])
        graphing_options.set_labels(xlabel='Index')
        plt.scatter(indices, y, marker='.')
        plt.show()
        
        hist, bins = np.histogram(y_trimmed, bins=20)
        plt.bar(bins[:-1], hist, width = bins[1]-bins[0])
        plt.ylim(0, 1.2 * np.max(hist))
        plt.xlabel(f'Raw {graphing_options.y_label} Value ({graphing_options.y_units})')
        plt.ylabel('Number of Occurences')
        plt.show()
    
    match method:
        case "digital":
            digital = (np.max(y_trimmed) - np.min(y_trimmed)) / (2 * np.sqrt(3))
            print('Digital Uncertainty:', digital)
            return digital
        case "default":
            return isolate_noise_uncertainty(data_trimmed, indices_range, y_range)
        case "manual":
            print('Manual Uncertainty:', manual_uncert)
            return manual_uncert


def isolate_noise_uncertainty(raw_data, indices_range, y_range):
    x, y, indices = raw_data
    
    y_ave = np.mean(y[indices_range[0]:indices_range[1]])
    y_std = np.std(y[indices_range[0]:indices_range[1]])
    
    print('Mean = ', y_ave,y_std)
    print('Standard Deviation (Noise Value) = ', y_std)
    
    return y_std

def pack_data(data, uncertainty, p=100, trim_range=None, save=False, plot=False, graphing_options=None):
    def pack(A, p):
        # A is an array, and p is the packing factor
        B = np.zeros(len(A)//p)
        i = 1
        while i - 1 < len(B):
            B[i-1] = np.mean(A[p*(i-1):p*i])
            i += 1
        return B
    
    x_raw, y_raw = data
    x = pack(x_raw, p)
    y = pack(y_raw, p)
    
    length = len(x)
    indices  = np.arange(length)
    x_uncert = np.zeros(length)
    
    y_uncert_raw  = uncertainty
    y_uncert_mean = y_uncert_raw / np.sqrt(p)
    y_uncert      = np.array([y_uncert_mean] * length)

    if trim_range is not None:
        x, x_uncert, y, y_uncert = \
            map(lambda a: a[trim_range[0]:trim_range[1]],
                (x, x_uncert, y, y_uncert))
        indices = np.arange(0, len(x))
        
    if plot:        
        plt.figure()
        graphing_options.plot_data(indices, y, y_uncert)
        graphing_options.set_labels()
        plt.xlabel('Index')
        plt.title('Packed Data')
        plt.show()

    if save:
        header = [np.array(['Time',  'u[time]', 'Voltage', 'u[Voltage]']), 
                  np.array(['(sec)', '(sec)',   '(V)',     '(V)'])]
        df = pd.DataFrame(np.array([x, x_uncert, y, y_uncert]).transpose(), columns=header)   
        
        csv_data = df.to_csv(output_name, index = False)
        print('Packed Data Stored in ', output_filename)
    
    return (x, x_uncert, y, y_uncert)

def calculate_chi_squared(fit_function, fit_params, x, y, sigma):
    dof = len(x) - len(fit_params)
    return np.sum((y - fit_function(x, *fit_params)) ** 2 / sigma ** 2) / dof

def calculate_t_score(a, da, b, db):
    return np.abs(a - b) / np.sqrt(da ** 2 + db ** 2)

@dataclass
class Dataset:
    x: np.array
    dx: np.array
    y: np.array
    dy: np.array

def autofit_dataset(dataset, graphing_options, initial_fit_parameters):
    return autofit((dataset.x, dataset.dx, dataset.y, dataset.dy), 
                   graphing_options,
                   initial_fit_parameters)

# +
def autofit_without_packing(x, y, y_uncert, 
                            graphing_options, initial_fit_parameters):
    return autofit((x, None, y, y_uncert), 
                   graphing_options,
                   initial_fit_parameters)

def autofit(packed_data: tuple,
            graphing_options: GraphingOptions,
            initial_fit_parameters: FitParameters):

    results = FitModelResult()
    
    fit_function = initial_fit_parameters.fit_function
    guesses = initial_fit_parameters.values()
    
    x, x_uncert, y, y_uncert = packed_data
    
    # Define 500 points spanning the range of the x-data; for plotting smooth curves
    x_theory = np.linspace(min(x), max(x), 500)

    # Compare the guessed curve to the data for visual reference
    y_guess = fit_function(x_theory, *guesses)
    
    plt.figure()
    graphing_options.plot_data(x, y, y_uncert, label='Measured Data');
    graphing_options.plot_model(x_theory, y_guess, initial_fit_parameters);
    graphing_options.set_labels();
    plt.title('Initial Parameter Guess')
    plt.legend(loc="best", numpoints=1)
    results.initial_guess_graph = graphing_options.save_graph_and_close()

    # calculate the value of the model at each of the x-values of the data set
    y_fit = fit_function(x, *guesses)
    
    # Residuals are the difference between the data and theory
    residual = y - y_fit

    # Plot the residuals
    plt.figure()
    graphing_options.plot_residuals(x, residual, y_uncert);
    graphing_options.set_labels()
    plt.ylabel(f"Residual y-y_fit [{graphing_options.y_units}]")
    plt.title("Residuals of Initial Parameter Guess")
    results.initial_guess_residuals_graph = graphing_options.save_graph_and_close()

    fit_params, fit_cov = curve_fit(
        fit_function, x, y, sigma=y_uncert, 
        p0=initial_fit_parameters.values(),
        absolute_sigma=True,
        maxfev=10**5)
    fit_params_error = np.sqrt(np.diag(fit_cov))
    
    match initial_fit_parameters:
        case SinusoidalFitParameters():
            results.parameters = SinusoidalFitParameters(
                amplitude = fit_params[0],
                frequency = fit_params[1],
                phase     = fit_params[2],
                
                amplitude_uncert = fit_params_error[0],
                frequency_uncert = fit_params_error[1],
                phase_uncert     = fit_params_error[2]
            )
            
        case OffsetSinusoidalFitParameters():
            results.parameters = OffsetSinusoidalFitParameters(
                amplitude = fit_params[0],
                frequency = fit_params[1],
                phase     = fit_params[2],
                offset    = fit_params[3],
                
                amplitude_uncert = fit_params_error[0],
                frequency_uncert = fit_params_error[1],
                phase_uncert     = fit_params_error[2],
                offset_uncert    = fit_params_error[3]
            )
            
        case ExponentialFitParameters():
            results.parameters = ExponentialFitParameters(
                amplitude     = fit_params[0],
                time_constant = fit_params[1],
                offset        = fit_params[2],
                
                amplitude_uncert     = fit_params_error[0],
                time_constant_uncert = fit_params_error[1],
                offset_uncert        = fit_params_error[2]
            )
        case RingdownFitParameters():
            results.parameters = RingdownFitParameters(
                amplitude          = fit_params[0],
                time_constant      = fit_params[1],
                resonant_frequency = fit_params[2],
                phase              = fit_params[3],
                
                amplitude_uncert          = fit_params_error[0],
                time_constant_uncert      = fit_params_error[1],
                resonant_frequency_uncert = fit_params_error[2],
                phase_uncert              = fit_params_error[3]
            )
        case LinearFitParameters():
            results.parameters = LinearFitParameters(
                slope     = fit_params[0],
                intercept = fit_params[1],
                
                slope_uncert     = fit_params_error[0],
                intercept_uncert = fit_params_error[1]
            )
        case LinearFitParametersOne():
            results.parameters = LinearFitParametersOne(
                slope     = fit_params[0],
                
                slope_uncert     = fit_params_error[0],
            )            
        case RCResponseFitParameters():
            results.parameters = RCResponseFitParameters(
                amplitude      = fit_params[0],
                time_constant  = fit_params[1],
                offset         = fit_params[2],
                
                amplitude_uncert          = fit_params_error[0],
                time_constant_uncert      = fit_params_error[1],
                offset_uncert = fit_params_error[2],
            )
        case LRCResponseFitParameters():
            results.parameters = LRCResponseFitParameters(
                amplitude          = fit_params[0],
                gamma              = fit_params[1],
                resonant_frequency = fit_params[2],

                amplitude_uncert          = fit_params_error[0],
                gamma_uncert              = fit_params_error[1],
                resonant_frequency_uncert = fit_params_error[2]
            )
            
    results.chi2 = calculate_chi_squared(fit_function, fit_params, x, y, y_uncert)
    results.covariance_matrix = fit_cov

    x_fitfunc = np.linspace(min(x), max(x), len(x))
    y_fitfunc = fit_function(x_fitfunc, *fit_params)
    y_fit = fit_function(x, *fit_params)
    residual = y-y_fit

    plt.figure()
    graphing_options.plot_data(x, y, y_uncert, label='Measured Data');
    graphing_options.plot_model(x_fitfunc, y_fitfunc, results.parameters);
    graphing_options.set_labels()
    plt.title('Best Fit of Function to Data')
    plt.legend(loc='best',numpoints=1)
    results.autofit_graph = graphing_options.save_graph_and_close()

    fig = plt.figure(figsize=(7,10))
    
    # The residuals plot
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.errorbar(x, residual, yerr=y_uncert,
                 marker='.', 
                 linestyle='', 
                 label="Residual (y-y_fit)")
    ax1.hlines(0,np.min(x),np.max(x),lw=2,alpha=0.8,color=graphing_options.model_color)
    ax1.set_xlabel(f"{graphing_options.x_label} [{graphing_options.x_units}]")
    ax1.set_ylabel(f"y-y_fit [{graphing_options.y_units}]")
    ax1.set_title('Residuals for the Best Fit')
    ax1.legend(loc='best',numpoints=1)

    # Histogram of the residuals
    ax2 = fig.add_subplot(2, 1, 2)
    hist,bins = np.histogram(residual,bins=30)
    ax2.bar(bins[:-1],hist,width=bins[1]-bins[0])
    ax2.set_ylim(0,1.2*np.max(hist))
    ax2.set_xlabel(f"y-y_fit [{graphing_options.y_units}]")
    ax2.set_ylabel('Number of occurences')
    ax2.set_title('Histogram of the Residuals')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    results.autofit_residuals_graph = Image.open(buf)
    plt.close()

    return results
