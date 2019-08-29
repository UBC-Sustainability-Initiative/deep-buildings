from datetime import timedelta
import numpy as np
import pandas as pd

def generate_data(X, y, freq: str, regr_vars=None, look_back=None, look_ahead=None, 
                  multiplier=None, baseline=None):
    '''
    
    freq: either 'D' or 'H'
    
    look_back: Values above must be integers. Determines how many previous days
               of data of each variable are including in the training/testing 
               variable set.
    
    look_ahead: Values above must be integers. Determines how many future days
               of data of each variable are including in the training/testing 
               variable set.
               
    multiplier:  
             1 = apply no adjustment to new_value, and calculate daily sum of 
                 all hourly values
             2 = take square of new_value, such that new_value = new_value ^ 2,
                 and calculate daily sum of all values
          -999 = take daily mean of new_value
          -888 = take daily mean, maximum, and minumum of new_value
          -777 = take daily mean, maximum of new_value
          
    baseline: 
        Values above must be integers or zero. The code will adjust any variable
        based on the following equation: new_value = original_value - baseline_value
        

        
    # Scenario 1: Assumes the target variable is indoor air temperature. The goal of the classifier
    # is to determine the number of days per summer that the ASHRAE Adaptive Comfort thermal comfort
    # criteria for naturally ventilated buildings (i.e., CIRS) is exceeded due to hot indoor conditions
    
    # Scenario 2: Assumes the target variable is indoor air temperature. The goal of the classifier
    # is to determine the number of days per summer that indoor temperatures exceed 27 deg C, which is 
    # colloquially considered 'hot'
    
    '''
    
    if regr_vars:
        regression_variables = regr_vars
        
    else: 
        regression_variables = list(X.columns)

    if look_back:
        look_back = look_back
    else: 
        look_back = np.zeros(len(regression_variables))
        
    if look_ahead:
        look_ahead = look_ahead
    else: 
        look_ahead = np.zeros(len(regression_variables))
        
    if multiplier:
        multiplier = multiplier
    else:
        multiplier = np.zeros(len(regression_variables))
        
    if baseline: 
        baseline_values = baseline 
    else: 
        baseline_values = np.zeros(len(regression_variables))
        
        
    regression_variables_handle= np.ones(len(regression_variables))
    # 0 - use actual (assumes value does not change daily)
    # 1 - use daily mean value
    # 2 - use daily mean, max, and min
    
    target_data_handle=[0,27]
    # 0 - use daily mean
    # 1 - use daily value above threshold, shown after comma
    # 2 - use daily maximum
    # 3 - use daily minimum
    
    if freq == 'D':
      Xd = X.resample('D').mean()
      
      for i in range(len(regression_variables_handle)):
        if regression_variables_handle[i]==1:
          Xd[regression_variables[i]]=X[regression_variables[i]].resample('D').mean()
        if regression_variables_handle[i]==2:
          mean_str=regression_variables[i]
          max_str=regression_variables[i]+'_max'
          min_str=regression_variables[i]+'_min'
          Xd[mean_str]=X[regression_variables[i]].resample('D').mean()
          Xd[max_str]=X[regression_variables[i]].resample('D').max()
          Xd[min_str]=X[regression_variables[i]].resample('D').min()
      
      X=Xd
    
      if target_data_handle[0]==0:
        yd=y.resample('D').mean()
      if target_data_handle[0]==1:
        yd=y.resample('D').mean()
        yd.values[:]=0
        yadd=y[y>target_data_handle[1]].resample('D').count()
        yd=yd+yadd
        yd=yd.fillna(0)
      if target_data_handle[0]==2:
        yd=y.resample('D').max()
      if target_data_handle[0]==3:
        yd=y.resample('D').min()
        
      y=yd
      
    X_historyKeys=['hum_ratio','hours','solar_radiation','temp','wind_dir',
                   'windspeed','L3S_Office_1']
    
    X_NhistorySteps=[2,2,2,2,2,2,2] # 3 or above is broken!
    # 1 = Assign rolling average 
    # 2 = Assign only value of last time step
    # 3 = Assign value of last time step + 1st-order derivative of last time step
    # 4 = Assign value of last time step + 1st-order derivative of last time step + 2nd-order derivative of last time step
    
    X_dropCurrent=[0,0,0,0,0,0,0]
    # 0 = Keeps present time-step value from training/test data
    # 1 = Drops present time-step value from training/test data
    
    for i in range(len(X_historyKeys)):
      for j in range(X_NhistorySteps[i]):
        header_str='next_'+X_historyKeys[i]+'_'+str(j)
        if j==0:
          if X_NhistorySteps[i]>1:
            continue
          else:
            X[header_str]=X[X_historyKeys[i]].rolling(window=24).mean()
            break
        if j==1: # Assign value of last time step
          X[header_str]=X[X_historyKeys[i]].shift(-4)
        if j==2: # Assign 1st order derivative of last time step
          X[header_str]=X[X_historyKeys[i]].shift(2)
        if j==3: # Assign 2nd order derivative of last time step
          X[header_str]=X[X_historyKeys[i]].shift(3)
      if X_dropCurrent[i]==1:
        X=X.drop(columns=[X_historyKeys[i]], axis=1)
          
    X['target_data']=y
    X = X.dropna()
    y=X['target_data']
    

    X = X.drop(columns=['target_data'], axis=1)
    
    X=X.iloc[2:]
    y=y.iloc[2:]
    
    return X, y