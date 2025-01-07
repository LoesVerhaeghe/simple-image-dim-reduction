import pandas as pd
import numpy as np
from utils.helpers import extract_SBH_columns 
from utils.plotting_utilities import plot_SBH, plot_SBH_with_yfit, plot_vhs_vs_TSS_with_fct, plot_parameter_with_conf_intervals
from src.parameter_estimation.batch_curve_processing import get_velocities_and_yfits
from src.parameter_estimation.parameter_estimation import get_parameters, SSE, MSE, MAE, MLE_Norm
from src.parameter_estimation.velocity_functions import vesilind, takacs, takacs_rP_cte
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
func=takacs_rP_cte #define function
folder_name='takacs' #define folder to save results (in results/parameter_estimation_results/..)
all_estimated_parameters = pd.DataFrame(columns=['V0', 'rH']) # define the name of the params
init_parameters=[19.75, 0.576] # define initial paramaters (m/h, l/g)
optim_method='L-BFGS-B'
param_estim_min_func=SSE
###define parameters for GLUE
threshold_HS=6
threshold_behavioural=0.80
conf_interval_size=0.95
lower_threshold=[0,0]
higher_threshold=[120,50]
number_of_repetitions= 100000 #best take 100 000 for good uncertainty intervals

for i, sheet_name in enumerate(all_sheetnames):
   # Get TSS and velocities
   extracted_data=extract_SBH_columns(file_name_batch_curves, sheet_name)
   velocities, yfits=get_velocities_and_yfits(extracted_data)   
   TSS=np.array(extracted_data.columns)

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

   conf_interval=GLUE_conf_interval(behavioural_likelihoods=behavioural_likelihoods, 
                                            behavioural_parameter_sets=behavioural_parameter_sets, 
                                            confidence=conf_interval_size)
   conf_intervals.append(conf_interval)

   ## parameter estimation
   init_parameters= np.median(conf_interval, axis=1)
   estimated_parameters=get_parameters(model_func=func, 
                             TSS=TSS, 
                             vhs=velocities, 
                             thetas_init=init_parameters, 
                             minim_fun=param_estim_min_func,
                             method=optim_method) 
   
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

# plot estimated parameters in a timeseries plot
dates = pd.to_datetime(all_sheetnames)
for i, column in enumerate(all_estimated_parameters):
   plot_parameter_with_conf_intervals(dates=dates, 
                                   parameter_values=all_estimated_parameters[column], 
                                   conf_intervals=[interval[i,:] for interval in conf_intervals], 
                                   parameter_name=column,
                                   file_path=f'results/parameter_estimation_results/{folder_name}/parameter_timeseries_{column}')
   

### save final parameters + conf intervals in excel file
# conf_intervals_numpy=np.array(conf_intervals)

# # Extract confidence intervals 
# V0_lower = conf_intervals_numpy[:, 0, 0]  # Lower bounds for V0
# V0_upper = conf_intervals_numpy[:, 0, 1]  # Upper bounds for V0
# rH_lower = conf_intervals_numpy[:, 1, 0]  # Lower bounds for rH
# rH_upper = conf_intervals_numpy[:, 1, 1]  # Upper bounds for rH

# # Add these intervals as columns in the dataframe
# all_estimated_parameters['V0_lower'] = V0_lower
# all_estimated_parameters['V0_upper'] = V0_upper
# all_estimated_parameters['rH_lower'] = rH_lower
# all_estimated_parameters['rH_upper'] = rH_upper

# all_estimated_parameters.to_excel('estimated_parameters.xlsx', index=False)