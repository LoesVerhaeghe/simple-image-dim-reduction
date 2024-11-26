import matplotlib.pyplot as plt
import numpy as np
import os 

def plot_SBH(df, sheetname, path, overwrite=False):
    """
    Plot DataFrame with time index.
    
    Parameters:
    df (pd.DataFrame): DataFrame with time index and numerical columns.
    sheetname (str): Title of the plot.
    path (str): Directory to save the plot.
    overwrite (bool): Whether to overwrite the file if it already exists.
    """
    file_path = f'{path}/{sheetname}.jpg'
    
    # Check if the file exists and if overwrite is False
    if not overwrite and os.path.exists(file_path):
        return  # Skip saving the file if it already exists and overwrite is False

    plt.figure(figsize=(6, 4), dpi=150)
    for column in df.columns:
        plt.plot(df.index, df[column], label=column)

    plt.xlabel('Time (min)')
    plt.ylabel('SBH')
    plt.title(sheetname)
    plt.legend(title='Init. MLSS concentration (g/L)')
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

def plot_SBH_with_yfit(df, yfit, sheetname, path, overwrite=False):
    """
    Plot DataFrame with time index.
    
    Parameters:
    df (pd.DataFrame): DataFrame with time index and numerical columns.
    title (str): Title of the plot.
    path (str): Directory to save the plot.
    overwrite (bool): Whether to overwrite the file if it already exists.
    """
    file_path = f'{path}/{sheetname}.jpg'
    
    # Check if the file exists and if overwrite is False
    if not overwrite and os.path.exists(file_path):
        return  # Skip saving the file if it already exists and overwrite is False

    plt.figure(figsize=(6, 4), dpi=150)
    for i, column in enumerate(df.columns):
        plt.plot(df.index, df[column], label=column)
        plt.plot(df.index, yfit[i], linestyle='dashed', color=plt.gca().lines[-1].get_color())

    plt.xlabel('Time (min)')
    plt.ylabel('SBH')
    plt.xlim([0,30])
    plt.ylim([0,1050])
    plt.title(sheetname)
    plt.legend(title='Init. MLSS concentration (g/L)')
    plt.grid(True)
    plt.savefig(file_path)
    plt.close()

def plot_vhs_vs_TSS_with_fct(velocities, TSS, x_plot, yv, params, path, plot_title, overwrite=False):
    file_path = f'{path}/{plot_title}.jpg'
    # Check if the file exists and if overwrite is False
    if not overwrite and os.path.exists(file_path):
        return  # Skip saving the file if it already exists and overwrite is False

    velocities=np.array(velocities)

    plt.figure(figsize=(6, 4), dpi=150)
    plt.scatter(TSS, velocities, label='Measurements') # plot measured vhs versus TSS
    plt.plot(x_plot, yv, label='Estimated function') # plot estimated function
    plt.xlabel('MLSS (g/L)')
    plt.ylabel(f'$v_{{hs}}$ (m/h)')
    plt.ylim([0,12])
    plt.legend()
    plt.title(plot_title + ' with params: '+', '.join([str(round(param, 3)) for param in params]))
    plt.savefig(file_path)
    plt.close()

def plot_parameter_with_conf_intervals(dates, parameter_values, conf_intervals, parameter_name, file_path):
    # Extract confidence intervals
    lower_bounds = [interval[0] for interval in conf_intervals]
    upper_bounds = [interval[1] for interval in conf_intervals]

    # Create the plot
    plt.figure(figsize=(10, 3), dpi=150)
    plt.plot(dates, parameter_values, '-o', label=f'{parameter_name}')
    plt.fill_between(dates, lower_bounds, upper_bounds, color='lightblue', alpha=0.9)
    plt.xlabel('Time')
    plt.ylabel(parameter_name)
    plt.grid(True)
    plt.savefig(file_path)

