import numpy as np
from scipy.optimize import minimize, Bounds
from src.parameter_estimation.velocity_functions import *

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

def Likelihood(fun, thetas, xdata, observed):
    y_mod= fun(thetas, xdata)
    weight_diff = (y_mod - observed) * 1 # weight can be set to one since no high concentrations are present anyways
    sumsquare = np.sum(weight_diff**2)
    L = np.abs(np.exp(-sumsquare))
    return L

def get_parameters(model_func, TSS, vhs, thetas_init, minim_fun, method):
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

    # Minimization to find optimal parameters (thetas)
    result = minimize(lambda thetas: minim_fun(model_func, thetas, TSS, vhs), thetas_init, method=method)

    # Extracting the optimized parameters
    optimized_thetas = result.x
    
    return optimized_thetas


# Parameter estimation vesilind
def get_parameters_vesilind(TSS, vhs, thetasV_init, minim_fun):
    """
    Function that estimates parameters of the Vesilind function
    Inputs
    TSS: g/L
    vhs: m/h
    thetasV_init: initial values for parameters v0 (m/h) and rv (l/g)
    minim_fun=function that needs to be minimized, can be SSE, MSE

    Outputs: 
    v0: m/h - the maximum settling velocity
    rv: l/g - model parameter
    """

    # Minimazation of SSE to find optimal parameters (thetas)
    result = minimize(lambda thetas: minim_fun(vesilind, thetas, TSS, vhs), thetasV_init, method='Nelder-Mead')

    # Extracting the optimized parameters
    thetasV = result.x

    par_V=[thetasV[0], thetasV[1]]  #m/h, l/g
    return par_V

def get_parameters_takacs(TSS, vhs, thetasT_init):
    """
    Function that estimates parameters of the Takacs function
    Inputs
    TSS: g/L
    vhs: m/h
    thetasV_init: initial values for parameters v0 (m/h), rh (l/g) and rp (l/g)

    Outputs: 
    v0: m/h - the maximum settling velocity
    rh: l/g - the settling characteristic of the hindered settling zone 
    rp: l/g - the settling characteristic at low solids concentrations
    """
    # Set upper and lower bounds for each parameter -> this was needed because optimal parameters became negative without bounds
    lower_bound = [0, 0, 0]  
    upper_bound = [np.inf,np.inf,np.inf]  

    # Create bounds for optimization
    bounds = Bounds(lower_bound, upper_bound)

    # Minimazation of SSE to find optimal parameters (thetas)
    result = minimize(lambda thetas: SSE(takacs, thetas, TSS, vhs), thetasT_init, method='L-BFGS-B', bounds=bounds)

    # Extracting the optimized parameters
    thetasT = result.x

    par_T=[thetasT[0], thetasT[1], thetasT[2]]  #m/h, l/g, l/g
    return par_T
