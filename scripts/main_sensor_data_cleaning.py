import pandas as pd
import matplotlib.pyplot as plt

# Initialize an empty dictionary to hold the data
data_dict = {}

df = pd.read_csv('data/sensor_data/MLSS_basin5.csv')
df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
df.set_index(df.columns[0], inplace=True)
data_dict['MLSS_basin5 [mg/L]'] = df

df = pd.read_csv('data/sensor_data/TSS_RAS.csv')
df[df.columns[0]] = pd.to_datetime(df[df.columns[0]])
df.set_index(df.columns[0], inplace=True)
data_dict['TSS_RAS [mg/L]'] = df

# Combine all DataFrames into a single DataFrame by concatenating them on columns
combined_df = pd.concat(data_dict.values(), axis=1)
combined_df = combined_df[combined_df.index >= '2023-10-01']
print(combined_df)



############### apply very basic data cleaning method:

# apply basic boundaries to values:
combined_df = combined_df[(combined_df['TSS_RAS'] >= 350) & (combined_df['TSS_RAS'] <= 12000)]
combined_df = combined_df[(combined_df['MLSS_basin5'] >= 100) & (combined_df['MLSS_basin5'] <= 5000)]


def clean_and_smooth(df, column, window_size=3, method='mean'):
    """
    Cleans and smooths the specified column in the dataframe.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column (str): The column to be cleaned and smoothed.
    window_size (int): The window size for the smoothing (default is 3).

    Returns:
    pd.DataFrame: The dataframe with the cleaned and smoothed column.
    """
    # Step 1: Fill missing values (NaN) with the column mean or any other method
    df[column] = df[column].fillna(df[column].mean())

    # Step 2: Apply smoothing using a moving average 
    df[column] = df[column].rolling(window=window_size, min_periods=1).mean()
    
    return df

smoothed_df=combined_df.copy()

smoothed_df = clean_and_smooth(smoothed_df, 'TSS_RAS', window_size=10, method='mean')
smoothed_df = clean_and_smooth(smoothed_df, 'MLSS_basin5', window_size=10, method='mean')

fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax[0].plot(combined_df.index, combined_df['MLSS_basin5'], label='MLSS_basin5 [mg/L]', color='blue')
ax[0].plot(smoothed_df.index, smoothed_df['MLSS_basin5'], label='MLSS_basin5 smoothed [mg/L]', color='red')
ax[0].set_ylabel('MLSS [mg/L]')
ax[0].legend(loc='upper right')

ax[1].plot(combined_df.index, combined_df['TSS_RAS'], label='TSS_RAS [mg/L]', color='blue')
ax[1].plot(smoothed_df.index, smoothed_df['TSS_RAS'], label='TSS_RAS smoothed [mg/L]', color='red')
ax[1].set_ylabel('TSS [mg/L]')
ax[1].legend(loc='upper right')

ax[1].set_xlabel('Time')


#resample to every 10 min
combined_df_10min = smoothed_df.resample('10T').first()

# Convert the index from datetime to days, starting from 0
combined_df_10min['days'] = (combined_df_10min.index - combined_df_10min.index[0]).total_seconds() / (3600 * 24)
combined_df_10min.set_index('days', inplace=True)


# look at final results
fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax[0].plot(combined_df_10min.index, combined_df_10min['MLSS_basin5'], label='MLSS_basin5 [mg/L]', color='blue')
ax[0].set_ylabel('MLSS [mg/L]')
ax[0].legend(loc='upper right')
ax[1].plot(combined_df_10min.index, combined_df_10min['TSS_RAS'], label='TSS_RAS [mg/L]', color='blue')
ax[1].set_ylabel('TSS [mg/L]')
ax[1].legend(loc='upper right')
ax[1].set_xlabel('Time')
plt.tight_layout()
plt.show()

#save results
combined_df_10min.to_csv('data/input_settler_model_smoothed.csv', sep='\t')

df_effluent = pd.read_csv('data/lab_measurements/lab_measurements_TSSeffluent.csv')
df_effluent[df_effluent.columns[0]] = pd.to_datetime(df_effluent[df_effluent.columns[0]])
df_effluent.set_index(df_effluent.columns[0], inplace=True)


df_effluent['time'] = (df_effluent.index - combined_df.index[0]).total_seconds() / (3600 * 24)
df_effluent.set_index('time', inplace=True)

df_effluent[0:].to_csv('data/lab_measurements/lab_measurements_TSSeffluent_west_index.csv', sep='\t')
