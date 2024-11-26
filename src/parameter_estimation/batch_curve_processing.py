import numpy as np
import pandas as pd

def slope_batch_curves(SBH, time):
    """
    Function that calculates the steepest slope of the settling curve by calculating the slope of every 3(+) min interval in the data.
    
    """
    Y = pd.to_numeric(SBH.values)
    time=pd.to_numeric(time)
    steepest = -1 #initiate a steepest slope, steepest slope cannot be lower than -1
    
    indices_intervals = {
    'ind1': [0,2,4,5,6,7,8,10,11,12,13],
    'ind2': [5,6,7,8,9,10,11,12,13,14,15]
    }


    indices_intervals = pd.DataFrame(indices_intervals)

    for j in range(0,len(indices_intervals)): 
        ind1=indices_intervals['ind1'][j]
        ind2=indices_intervals['ind2'][j]
        coeffs = np.polyfit(time[ind1:ind2], Y[ind1:ind2], deg=1) #linear least squares polynomial fit to the data points
        slope = abs(coeffs[0]) #extracts the slope of the linear fit
        
        if slope > steepest: #if sloper is steeper than previous slope (and slope should be higher than -1)
            steepest = slope #save new slope
            mid_interval = int((ind1+ind2-1) / 2) #save middle of the interval where steepest slope appears
    
    velocity = steepest # hindered settling velocity is the steepest slope of the settling curve
    yfit = -velocity * (time - time[mid_interval]) + Y[mid_interval] #linear hindered velocity function to plot
    
    return velocity, yfit


def get_velocities_and_yfits(dataframe):
    """
    Function that calculates the velocities (m/h) and yfits for all columns in a dataframe.
    """
    velocities=[]
    yfits=[]
    for i, column_name in enumerate(dataframe.columns):
        v, y = slope_batch_curves(dataframe[column_name], dataframe.index)
        #mL/min to m/h: *60/(100*(1000/28))
        #mL->cm = /(1000/28) and cm->m /100 and /min->/h-> * 60
        velocities.append(v*60/(100*(1000/28))) #m/h
        yfits.append(y)

    return velocities, yfits