import pandas as pd
import numpy as np
from utils.helpers import extract_SBH_columns 
from utils.plotting_utilities import plot_SBH, plot_SBH_with_yfit, plot_vhs_vs_TSS_with_fct, plot_parameter_with_conf_intervals
from src.parameter_estimation.batch_curve_processing import get_velocities_and_yfits
from src.parameter_estimation.parameter_estimation import get_parameters, SSE
from src.parameter_estimation.velocity_functions import vesilind, takacs
from src.parameter_estimation.GLUE import GLUE_all_params, plot_likelihood_vs_params, GLUE_behavioural, GLUE_conf_interval

# plot all curves + velocity and save all plots
file_name_batch_curves='data/settling_tests/Batch_settleability_test_settlingcurves.xlsx'
all_sheetnames=pd.ExcelFile(file_name_batch_curves).sheet_names

for sheet_name in all_sheetnames:
   # extract, plot the sludge blanket heights and save the plots
   extracted_data=extract_SBH_columns(file_name=file_name_batch_curves, 
                                      sheet_name=sheet_name)
   plot_SBH(df=extracted_data, 
            sheetname=sheet_name, 
            path='results/parameter_estimation_results/settling_curves',
            overwrite=False)

   # calculate the velocities and yfits, plot the sludge blanket heights with yfits and save the plots
   velocities, yfits=get_velocities_and_yfits(dataframe=extracted_data)
   plot_SBH_with_yfit(df=extracted_data, 
                      sheetname=sheet_name, 
                      yfit=yfits,
                      path='results/parameter_estimation_results/settling_curves_w_velocity',
                      overwrite=False)   
   
# Define x-data at which the fitted functions will be calculated to make a plot
x_plot = np.arange(0, 10, 0.01)
conf_intervals=[]

###define parameters for parameter estimation:
func=takacs #define function
folder_name='takacs' #define folder to save results (in results/parameter_estimation_results/..)
all_estimated_parameters = pd.DataFrame(columns=['v0', 'rh', 'rp']) # define the name of the params
init_parameters=[300/24,0.57,0.5] # define initial paramaters, (vesilind=according to experimental methods in wastewater treatment book)
param_estim_min_func=SSE
optim_method='Nelder-Mead'

###define parameters for GLUE
threshold_HS=6
threshold_behavioural=0.5
conf_interval_size=0.95
lower_threshold=[0,0,0]
higher_threshold=[200,50,500]
number_of_repetitions= 100000 #best take 100000 for good uncertainty intervals

for i, sheet_name in enumerate(all_sheetnames):
   # Get TSS and velocities
   extracted_data=extract_SBH_columns(file_name_batch_curves, sheet_name)
   velocities, yfits=get_velocities_and_yfits(extracted_data)   
   TSS=np.array(extracted_data.columns)
   
   # Estimate parameters and save plots of fitted functions
   estimated_parameters=get_parameters(model_func=func, 
                             TSS=TSS, 
                             vhs=velocities, 
                             thetas_init=init_parameters, 
                             minim_fun=param_estim_min_func,
                             method=optim_method) # first time, the init parameters are literature values, next, they are the parameters from the previous sampling date
   
   all_estimated_parameters.loc[i]=estimated_parameters 

   yv = func(estimated_parameters, x_plot) # Generating predicted values using the function (for the plot)
   
   plot_vhs_vs_TSS_with_fct(velocities=velocities, 
                            TSS=TSS, 
                            x_plot=x_plot, 
                            yv=yv, 
                            params=estimated_parameters, 
                            path=f'results/parameter_estimation_results/{folder_name}/fitted_functions/',
                            plot_title=sheet_name,
                            overwrite=True)
   
   init_parameters=estimated_parameters

   # Calculate GLUE confidence intervals
   likelihood_values, parameter_sets=GLUE_all_params(fun=func, 
                                                     TSS=TSS, 
                                                     Vhs=velocities, 
                                                     threshold_HS=threshold_HS, 
                                                     T_low=lower_threshold, 
                                                     T_high=higher_threshold, 
                                                     n=number_of_repetitions)
   
   plot_likelihood_vs_params(parameter_sets=parameter_sets, 
                             likelihoods=likelihood_values, 
                             threshold_behavioural=threshold_behavioural, 
                             path=f'results/parameter_estimation_results/{folder_name}/GLUE/', 
                             plot_title=sheet_name,
                             overwrite=True)
   
   behavioural_likelihoods, behavioural_parameter_sets = GLUE_behavioural(likelihoods=likelihood_values, 
                                                                          parameter_sets=parameter_sets, 
                                                                          threshold_behavioural=threshold_behavioural)

   conf_intervals.append(GLUE_conf_interval(behavioural_likelihoods=behavioural_likelihoods, 
                                            behavioural_parameter_sets=behavioural_parameter_sets, 
                                            confidence=conf_interval_size))

# plot estimated parameters in a timeseries plot
dates = pd.to_datetime(all_sheetnames)
for i, column in enumerate(all_estimated_parameters):
   plot_parameter_with_conf_intervals(dates=dates, 
                                   parameter_values=all_estimated_parameters[column], 
                                   conf_intervals=[interval[i,:] for interval in conf_intervals], 
                                   parameter_name=column,
                                   file_path=f'results/parameter_estimation_results/{folder_name}/parameter_timeseries_{column}')
