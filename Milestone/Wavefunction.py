import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats

import Variables as V
import Paths as P

np.random.seed(V.variable['Random seed'])

x_values = np.linspace(0, V.variable['complex time interval'], V.variable['spatial latice points'])
y_values, y_errors = P.run_path()

#Chi squared fitting functions
def deal_with_zero_errors(y_errors):
    #Function to deal with zero errors in the data.
    #Returns:    y_errors (array): The modified errors associated with the data.
    smallest_error = np.inf
    null_error = np.where(y_errors == 0)[0]
    for i in range(len(y_errors)):
        if y_errors[i] < smallest_error and y_errors[i] != 0:
            smallest_error = y_errors[i]
    if smallest_error == np.inf:
        smallest_error = 1e-15
    for i in null_error:
        y_errors[i] += smallest_error
    return y_errors
y_errors = deal_with_zero_errors(y_errors)

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
        x_values,
        y_values,
        sigma=y_errors,
        absolute_sigma=True, # errors bars DO represent 1 std error
        p0=initial_values, # starting point for fit
        check_finite=True) # raise ValueError if NaN encountered

    chi_squared_min = chi_squared(params, model_function, x_values, y_values, y_errors)
    degrees_of_freedom = x_values.size - params.size
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
            name = m
    print(f'The path is best modeled using a {name} fit')
    return params, params_errs, reduced_chi_squared, prob, name

print(y_values)
print(y_errors)
params, params_errs, reduced_chi_squared, prob, name  = run_models()
smooth_x = np.linspace(x_values[0], x_values[-1], 1000) # more points, over range of data
plt.plot(smooth_x, np.exp(-Models[name](smooth_x , *params)), color='r')
plt.plot(smooth_x, np.exp(-0.5*smooth_x**2), color='b')