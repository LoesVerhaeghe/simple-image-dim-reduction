import numpy as np
from scipy.optimize import minimize, Bounds, basinhopping
from src.parameter_estimation.velocity_functions import *
from scipy import stats


# Define sum of squared error function: will be used to fit the functions to the datapoint.
def SSE(fun, thetas, xdata, observed):
    """
    sum of squared errors
    """
    return np.sum((fun(thetas, xdata) - observed) ** 2)

def MSE(fun, thetas, xdata, observed):
    """
    mean squared errors
    """
    n = len(observed)
    return np.sum((fun(thetas, xdata) - observed) ** 2) /n

def MAE(fun, thetas, xdata, observed):
    """
    mean squared errors
    """
    n = len(observed)
    return np.sum(np.abs(fun(thetas, xdata) - observed)) /n

def MLE_Norm(fun, thetas, xdata, observed):
   y_mod=fun(thetas, xdata)
   res = observed-y_mod
   standard_dev = np.std(res)
   # Calculate the log-likelihood for normal distribution
   LL = np.sum(stats.norm.logpdf(observed, y_mod, standard_dev))
   # Calculate the negative log-likelihood
   neg_LL = -1*LL
   return neg_LL

def get_parameters(model_func, TSS, vhs, thetas_init, minim_fun, method='L-BFGS-B'):
    """
    Function that estimates parameters for a given model.
    
    Inputs:
    model_func: Function representing the settling model (e.g., Vesilind or others).
    TSS: g/L
    vhs: m/h
    thetas_init: Initial values for the model parameters (v0, rv, etc.).
    minim_fun: Function to minimize (e.g., SSE, MSE).

    Outputs:
    Optimized model parameters.
    """
    # Define bounds 
    bounds = [(0, None) for _ in thetas_init]  # This ensures all parameters are >= 0

    # Minimization to find optimal parameters (thetas)
    result = minimize(lambda thetas: minim_fun(model_func, thetas, TSS, vhs), thetas_init, method=method, bounds=bounds)

    # Extracting the optimized parameters
    optimized_thetas = result.x
    
    return optimized_thetas


def get_parameters_basin_hopping(model_func, TSS, vhs, thetas_init, minim_fun, method, niter=100, stepsize=0.5):
    """
    Function that estimates parameters for a given model using Basin Hopping.
    
    Inputs:
    model_func: Function representing the settling model (e.g., Vesilind or others).
    TSS: g/L
    vhs: m/h
    thetas_init: Initial values for the model parameters (v0, rv, etc.).
    minim_fun: Function to minimize (e.g., SSE, MSE).
    niter: Number of hopping iterations (default = 100).
    stepsize: Step size for random perturbations (default = 0.5).

    Outputs:
    Optimized model parameters.
    """

    # Define the local minimizer (you can choose BFGS, Nelder-Mead, etc.)
    minimizer_kwargs = {"method": method}

    # Objective function for Basin Hopping (minim_fun returns the error value to minimize)
    def objective_function(thetas):
        return minim_fun(model_func, thetas, TSS, vhs)

    # Basin Hopping algorithm for global optimization
    result = basinhopping(objective_function, thetas_init, niter=niter, minimizer_kwargs=minimizer_kwargs, stepsize=stepsize)

    # Extracting the optimized parameters
    optimized_thetas = result.x
    
    return optimized_thetas
