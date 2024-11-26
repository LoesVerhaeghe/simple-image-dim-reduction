import numpy as np
import matplotlib.pyplot as plt
import os
from src.parameter_estimation.velocity_functions import vesilind

def Likelihood(W, y_obs, y_mod):
    weight_diff = (y_mod - y_obs) * W
    sumsquare = np.sum(weight_diff**2)
    L = np.abs(np.exp(-sumsquare))
    return L

def GLUE_all_params(fun, TSS, Vhs, threshold_HS, T_low, T_high, n):
    '''
    Perform Generalized Likelihood Uncertainty Estimation (GLUE) to evaluate the likelihood  of model parameter sets given observed data.

    Parameters:
    -----------
    fun : function
        The function that computes model outputs (Should be either: diehl, cole, takacs, vesilind).
    TSS : numpy array
        Array of observed TSS concentrations.
    Vhs : numpy array
        Array of observed velocities corresponding to concentrations `conc`.
    threshold_HS : float
        Threshold concentration of TSS above which the weighting factor is 0.2; otherwise, it is 1.
    T_low : numpy array
        Array of lower bounds for model parameters.
    T_high : numpy array
        Array of upper bounds for model parameters.
    n : int
        Number of parameter sets (iterations) to generate and evaluate.

    Returns:
    --------
    likelihood_values : numpy array
        Array of likelihood values.
    parameter_sets : numpy array
        Array of parameter sets that correspond to the likelihood values.

    '''
    # if the TSS concentration is higher than threshold_HS then the weighting factor is 0.2. Else it is 1.
    W = np.zeros(len(TSS))
    for j in range(len(TSS)):
        if TSS[j] >= threshold_HS:
            W[j] = 0.2
        else:
            W[j] = 1

    parameter_sets = np.zeros((n, len(T_low)))
    likelihood_values = np.zeros(n)
    T_high=np.array(T_high)
    T_low=np.array(T_low)
    for i in range(n):
        randomnr = np.random.rand(len(T_low))
        thetas = T_low + (T_high - T_low) * randomnr #create random parameterset
        Vhs_estimate = fun(thetas, TSS) #calculate function (velocities) with this random parameterset
        likelihood_values[i] = Likelihood(W, Vhs, Vhs_estimate)
        parameter_sets[i, :] = thetas
    
    # Normalize likelihoods
    lmin = np.min(likelihood_values)
    lmax = np.max(likelihood_values)
    normalized_likelihoods = (likelihood_values - lmin) / (lmax - lmin)

    return normalized_likelihoods, parameter_sets

def GLUE_behavioural(likelihoods, parameter_sets, threshold_behavioural):
    # Filter out behavioural parameter sets
    behavioural_indices = likelihoods > threshold_behavioural
    behavioural_likelihoods = likelihoods[behavioural_indices]
    behavioural_parameter_sets = parameter_sets[behavioural_indices, :]
    return behavioural_likelihoods, behavioural_parameter_sets

def plot_likelihood_vs_params(parameter_sets, likelihoods, threshold_behavioural, path, plot_title, overwrite=False):
    """
    Plot a scatter plot of parameter sets vs. their likelihoods, with points colored based on a threshold.
    
    Parameters:
    -----------
    parameter_sets : numpy array
        Array of parameter sets.
    likelihoods : numpy array
        Array of behavioural likelihood values corresponding to the parameter sets.
    threshold_behavioural : float
        Likelihood threshold to determine the color of the scatter points.
    plot_title : str
        Title for the entire plot.
    """
    file_path=f'{path}/{plot_title}.jpg'
    # Check if the file exists and if overwrite is False
    if not overwrite and os.path.exists(file_path):
        return  # Skip saving the file if it already exists and overwrite is False
    
    num_parameters = parameter_sets.shape[1]
    
    # Define colors
    above_threshold_color = 'blue'
    below_threshold_color = 'navy'
    
    plt.figure(figsize=(5, 6))

    if plot_title:
        plt.suptitle(plot_title)  # Add general title for the entire plot
  
    for i in range(num_parameters):
        plt.subplot(num_parameters, 1, i + 1)
        plt.scatter(parameter_sets[:, i], likelihoods, 
                    c=np.where(likelihoods > threshold_behavioural, above_threshold_color, below_threshold_color),
                    s=5)  
        plt.axhline(y=threshold_behavioural, color='green', linestyle='--', linewidth=2)
        plt.xlabel(f'Parameter {i+1}')
        plt.ylabel('Likelihood')
    
    # Add legend
    threshold_line = plt.Line2D([0], [0], color='green', linestyle='--', linewidth=2, label='Behavioural threshold')
    plt.legend(handles=[threshold_line], loc='upper center', bbox_to_anchor=(0.5, -0.3), ncol=3)

    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

def GLUE_conf_interval(behavioural_likelihoods, behavioural_parameter_sets, confidence=0.95):
    """
    Calculate confidence intervals for each parameter in the parameter sets based on GLUE likelihoods.

    Parameters:
    -----------
    behavioural_likelihoods : numpy array
        Array of likelihood values corresponding to each parameter set.
    behavioural_parameter_sets : numpy array
        Array of parameter sets generated by GLUE (shape: [n, num_parameters]).
    confidence : float
        Confidence level for the intervals (default is 0.95 for a 95% confidence interval).

    Returns:
    --------
    confidence_intervals : numpy array
        Array of confidence intervals for each parameter (shape: [num_parameters, 2]).
        Each row contains the lower and upper bounds of the confidence interval for a parameter.
    """
    num_parameters = behavioural_parameter_sets.shape[1]
    lower_bound = (1.0 - confidence) / 2.0
    upper_bound = 1.0 - lower_bound
    
    confidence_intervals = np.zeros((num_parameters, 2))

    for i in range(num_parameters):
        # Sort parameter values and corresponding likelihoods
        sorted_indices = np.argsort(behavioural_parameter_sets[:, i])
        sorted_params = behavioural_parameter_sets[sorted_indices, i]
        sorted_likelihoods = behavioural_likelihoods[sorted_indices]
        
        # Calculate cumulative sum of normalized likelihoods
        cumulative_likelihoods = np.cumsum(sorted_likelihoods) / np.sum(sorted_likelihoods)
        
        # Interpolate to find parameter values at the specified confidence bounds
        lower_ci = np.interp(lower_bound, cumulative_likelihoods, sorted_params)
        upper_ci = np.interp(upper_bound, cumulative_likelihoods, sorted_params)
        
        confidence_intervals[i, 0] = lower_ci
        confidence_intervals[i, 1] = upper_ci
    return confidence_intervals