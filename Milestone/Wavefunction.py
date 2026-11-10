import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

import Variables as V
import Paths as P

np.random.seed(V.variable['Random seed'])

t_values = np.linspace(0, V.variable['complex time interval'], V.variable['spatial latice points'])
r_values, r_errors = P.run_path()
#even extension about t=0
displacement_values = np.append(r_values[::-1], r_values)
t_values = np.append(-t_values[::-1], t_values)

#Chi squared fitting functions
def deal_with_zero_errors(errors):
    #Function to deal with zero errors in the data.
    #Returns:    errors (array): The modified errors associated with the data.
    smallest_error = np.inf
    null_error = np.where(errors == 0)[0]
    for i in range(len(errors)):
        if errors[i] < smallest_error and errors[i] != 0:
            smallest_error = errors[i]
    if smallest_error == np.inf:
        smallest_error = 1e-15
    for i in null_error:
        errors[i] += smallest_error
    return errors
r_errors = deal_with_zero_errors(r_errors)
displacement_errors = np.append(r_errors[::-1], r_errors)

#Chi squared fitting functions
def chi_squared(model_params, model, x_data, y_data, y_err):
    return np.sum(((y_data - model(x_data, *model_params))/y_err)**2)

def path_model(model_function, initial_values):
    #Function to estimate the equation of the generated path.
    #Returns:    params (array): The optimized parameters of the model function.
    #            params_errs (array): The errors associated with the optimized parameters.
    #            reduced_chi_squared (float): The reduced chi-squared value of the fit.
    #            Prob (float): The probability of obtaining a chi-squared value as extreme as the observed value.
    params, cov = scipy.optimize.curve_fit(
        model_function,
        t_values,
        displacement_values,
        sigma=displacement_errors,
        absolute_sigma=True, # errors bars DO represent 1 std error
        p0=initial_values, # starting point for fit
        check_finite=True) # raise ValueError if NaN encountered

    chi_squared_min = chi_squared(params, model_function, t_values, displacement_values, displacement_errors)
    degrees_of_freedom = t_values.size - params.size
    reduced_chi_squared = chi_squared_min / degrees_of_freedom
    Prob = scipy.stats.chi2.sf(chi_squared_min, degrees_of_freedom)
    params_errs = np.sqrt(np.diag(cov))
    return params, params_errs, reduced_chi_squared.astype(float), Prob.astype(float)

#Define set of Models for fitting
def constant(x, *params):
    return params[0]
def linear (x, *params):
    return params[0] + params[1]*x
def quadratic (x, *params):
    return params[0] + params[1]*x + params[2]*x**2
def cubic (x, *params):
    return params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3
def quartic(x, *params):
    return params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*x**4

#Dictionary with model and number of variables
Model_variables = {'constant':1,
          'linear':2,
          'quadratic':3,
          'cubic':4,
          'quartic':5}
#Dictionary that links to model functions
Models = {'constant':constant,
          'linear':linear,
          'quadratic':quadratic,
          'cubic':cubic,
          'quartic':quartic}

def simpson_integration(function, model_name,  params):
    #Function to calculate the numerical integral using Simpson's method
    #Returns:   Area(float): total numerical value of the integration
    Area = 0
    ts = t_values[0]
    tf = t_values[-1]
    dt = (tf - ts) / V.variable['integration steps']
    for i in range(V.variable['integration steps']):
            t1 = ts + i*dt
            tm = ts + (i+1/2)*dt
            t2 = ts + (i+1)*dt
            Area += dt/6 * (function(model_name, t1, params) + 4*function(model_name, tm, params) + function(model_name, t2, params))
    return Area

def run_models():
    #Function to calculate the best model based on chi squared statistics.
    #Returns:    params (array): The optimized parameters of the model function.
    #            params_errs (array): The errors associated with the optimized parameters.
    #            reduced_chi_squared (float): The reduced chi-squared value of the fit.
    #            Prob (float): The probability of obtaining a chi-squared value as extreme as the observed value.
    reduced_chi_squared = np.inf
    for m, a in Model_variables.items():
        initial_params = np.random.rand(a)
        initial_params[0] = V.variable['start point']
        trial_params, trial_params_errs, trial_reduced_chi_squared, trial_Prob = path_model(Models[m], initial_params)
        #Minimise reduced chi squared without overfitting (<1).
        if (trial_reduced_chi_squared >= 1 and trial_reduced_chi_squared < reduced_chi_squared):
            params = trial_params
            params_errs = trial_params_errs
            reduced_chi_squared = trial_reduced_chi_squared
            prob = trial_Prob
            model_name = m
    return params, params_errs, reduced_chi_squared, model_name

def unnormalised_wavefunction(model_name, x, params):
    return (np.exp(1)**(-Models[model_name](x, *params)))**2

def normalisation_constant(model_name, params, params_errs):
    #Function to calculate the normalisation constant of the model function using Simsons Rule.
    #Returns:    norm_const (float): The normalisation constant of the model function.
    err_cont = np.zeros(len(params))
    norm_const = 2/simpson_integration(unnormalised_wavefunction, model_name, params)
    #error contributions (functional approach)
    for i in range(len(params)):
        params[i] += params_errs[i]
        err_cont[i] += (2/simpson_integration(unnormalised_wavefunction, model_name, params) - norm_const)**2
        params[i] -= params_errs[i]
    norm_err = np.sqrt(np.sum(err_cont))
    return norm_const, norm_err

def normalised_wavefunction(model_name, x, params, params_errs):
    norm_const, norm_err = normalisation_constant(model_name, params, params_errs)
    return norm_const*unnormalised_wavefunction(model_name, x, params)

params, params_errs, reduced_chi_squared, model_name  = run_models()
norm_const, norm_err = normalisation_constant(model_name, params, params_errs)
smooth_x = np.linspace(t_values[0], t_values[-1], 1000) # more points, over range of data

print(f'The path is best modeled using a {model_name} fit')
print(params)
plt.figure(1)
plt.scatter(t_values,displacement_values)
plt.plot(smooth_x, Models[model_name](smooth_x, *params))

plt.figure(2)
plt.plot(smooth_x, normalised_wavefunction(model_name, smooth_x, params, params_errs), color='r')
plt.plot(smooth_x, np.exp(-0.5*smooth_x**2)**2 * np.sqrt(1/np.pi), color='b')
plt.gca().set_ylim([0,0.8])
plt.gca().set_xlim([0,4])
plt.show()