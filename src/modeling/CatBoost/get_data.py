from datetime import timedelta
import numpy as np
import pandas as pd

def generate_data(df, freq: str, scenario=None, regr_vars = None, 
                  multiplier = None, 
                  baseline = None,
                  look_back = None,
                  look_ahead = None):
    '''
    
    freq: either 'D' or 'H'
    

    '''
    
    y = df['target_data']
    X = df.drop(columns=['target_data'])
    
    if regr_vars:
        regression_variables = regr_vars
    else: 
        regression_variables = list(X.columns)
    
    if multiplier: 
        multiplier = multiplier
    else: 
        multiplier = np.ones(len(regression_variables))
        
    if baseline: 
        baseline_values = baseline
        
    else:
        baseline_values = np.zeros(len(regression_variables))
    
    if look_back:
        look_back = look_back
    else: 
        look_back = np.zeros(len(regression_variables))
    
    if look_ahead:
        look_ahead = look_ahead
    else: 
        look_ahead = np.zeros(len(regression_variables))
    
    if freq == 'D':

      # Synchronize X and y data by ensuring consistency of timestamps and all
      # 'na' values dropped
      X['y_data']=y
      X = X.dropna()
      y=X['y_data']
      X = X.drop(columns=['y_data'], axis=1)
    
      Xp = X.copy() # Use 'Xp' as a temporary dataframe for processing to follow   
    
      for i in range(len(regression_variables)): # For each regression_variable
    
        if baseline_values[i]>0: # Apply shift in data values based on baseline_values list
          Xp[regression_variables[i]]=Xp[regression_variables[i]]-baseline_values[i]
    
        if multiplier[i]>=1: # Take daily mean of hourly data or daily mean-of-squared hourly data as per multiplier list
          if multiplier[i]==2:
            Xp[regression_variables[i]]=Xp[regression_variables[i]]**2
          Xp[regression_variables[i]]=Xp[regression_variables[i]].resample('D').sum()
    
        elif multiplier[i]==-999: # Take daily mean of hourly data
          Xp[regression_variables[i]]=Xp[regression_variables[i]].resample('D').mean()
    
        elif multiplier[i]==-888: # Take daily mean, max, and minimum of hourly data
          maxstr = regression_variables[i]+'_max'
          minstr = regression_variables[i]+'_min'
          regression_variables.append(maxstr)
          look_back.append(look_back[i])
          look_back.append(look_back[i])
          look_ahead.append(look_ahead[i])
          look_ahead.append(look_ahead[i])
          Xp[maxstr] = Xp[regression_variables[i]].resample('D').max()
          regression_variables.append(minstr)
          Xp[minstr] = Xp[regression_variables[i]].resample('D').min()
          Xp[regression_variables[i]]=Xp[regression_variables[i]].resample('D').mean()
    
        elif multiplier[i]==-777: # Take daily mean, and max of hourly data
          maxstr = regression_variables[i]+'_max'
          regression_variables.append(maxstr)
          look_back.append(look_back[i])
          look_ahead.append(look_ahead[i])
          Xp[maxstr] = Xp[regression_variables[i]].resample('D').max()
          Xp[regression_variables[i]]=Xp[regression_variables[i]].resample('D').mean()
    
      Xp=Xp.resample('D').sum() # Xp will not be fully resampled to daily values, so this cleans everything up.
    
      # Apply look ahead and look back values
    
      for i in range(len(regression_variables)):
        if look_back[i]>0:
          for x in range(look_back[i]):
            header_str = 'last_'+regression_variables[i]+'_'+str(x+1)
            Xp[header_str]=Xp[regression_variables[i]].shift(x+1)
        if look_ahead[i]>0:
          for x in range(look_ahead[i]):
            header_str = 'next_'+regression_variables[i]+str(x+1)
            Xp[header_str]=Xp[regression_variables[i]].shift(-x-1)

      X = Xp # Reframe X based on Xp
    
      if scenario == 1:
        
      # Apply ASHRAE Adaptive Comfort model criteria to classify hot days vs. non-hot days
      # The Adaptive Comfort model is a standard model used to predict the minimum and maximum indoor air temperatures
      # that are typically considered by building occupants to yield comfortable indoor conditions. It applies
      # only to naturally-ventilated buildings, which is fine as this is what CIRS is.
      # It is a simple model, whereby Threshold_indoor_temp = f(Outdoor_Temp) only
      # See the CBE thermal comfort tool for more info: http://comfort.cbe.berkeley.edu
      # 
      # For this iteration, we're using the equation that denotes the upper bound of the 90% accepability
      # limit for indoor temperature, meaning that 10% of typical building occupants will be uncomfortable 
      # at indoor levels above what's stated. We evaluate the equation based on mean monthly outdoor air temperature:
      # Threshold_operative_temperature = 0.31 * Mean_monthly_outdoor_T + 20.3 [all in deg C]
      #
      # The code below calculates how many days in a summer do indoor air temperature, at any hour of the day, exceed
      # the threshold temperature limit;
      #
      # As the adaptive comfort model is based on operative temperautre, and we do not yet have this knowledge in full,
      # we will assume that the daytime operative temperature in CIRS at peak hot-weather hours is 1 deg C above
      # the measured air temperature. This will be verified at a future date.
      
        ACupper = X['temp'].copy()
        ACupper=ACupper.resample('M').mean().resample('D').ffill().reset_index()
        ACupper=ACupper.set_index(pd.DatetimeIndex(ACupper['index']))
        ACupper=ACupper.drop(columns=['index'])
        ACupper=ACupper.multiply(0.33).add(20.8)
        ACexceed=0-ACupper.sub(y.resample('D').max(),axis=0)
        ACexceed[ACexceed<=0]=0
        ACexceed[ACexceed>0]=1
        yp = ACexceed
        
      elif scenario == 2:
        yp=y[y>27].resample('D').count()
        yp[yp>0]=1


      # Re-synchronize X and y data by ensuring consistency of timestamps and all 'na' values dropped
      X['y_data']=yp
      X=X.dropna()
      y=X['y_data']
      
#      if corr_plot: 
#          import matplotlib.pyplot as plt
#          import seaborn as sns
#          plt.figure()
#          sns.heatmap(X.corr())
#          X = X.dropna()
#          y=X['y_data']
#          # Print correlation values to help hint at what regression parameters to choose
#          CorrMatrix=pd.DataFrame(X.corr()['y_data'])
#          print(CorrMatrix.sort_values(by=['y_data']))
#    
      X = X.drop(columns=['y_data'], axis=1)   

    if freq == 'H':
      X_historyKeys=['solar_radiation','temp','wind_dir','hum_ratio','hours',
                     'windspeed']
    
      X_lookback=[6,24,4,4,0,2]#,2]
      X_lookahead=[0,2,2,0,0,0]#,2]
      X_drop=[0,0,0,0,0,0]
      for i in range(len(X_historyKeys)):
        for j in range(X_lookback[i]):
          header_str='last_'+X_historyKeys[i]+'_'+str(j+1)
          X[header_str]=X[X_historyKeys[i]].shift(j+1)
        for j  in range(X_lookahead[i]):
          header_str='next_'+X_historyKeys[i]+'_'+str(j+1)
          X[header_str]=X[X_historyKeys[i]].shift(-1-j)
        if X_drop[i]==1:
          X=X.drop(columns=[X_historyKeys[i]])
    
#      # Add in is weekend, rolling std features 
#      weekends = np.where(X.index.dayofweek-5>=0, 1, 0)
#      X['is_weekend'] = weekends
#      X['rolling_std_4'] = X['temp'].rolling(4).std()
#      X['rolling_std_3'] = X['temp'].rolling(3).std()
#      X['rolling_std_2'] = X['temp'].rolling(2).std()
#      X['rolling_std_mean_3'] = X['temp'].rolling(3).std().rolling(3).mean()
#      X['temp_gradient'] = np.gradient(X['temp'].values)
#    
#      # Add if previous value exceeds 25 degrees
#      X['future_exceedence'] = np.where(X['temp'].shift(-2)>=27, 1, 0)
#      X['prev_exceedence'] = np.where(X['temp'].shift(2)>=27, 1, 0)
#    
#      # Add last 3 hours of experienced indoor temperature
#    
#      X['hist_indoor'] = X['indoorTemp'].shift(3)
#      X['hist_indoor_diff'] = X['indoorTemp'].shift(-3).diff(2)
#    
#      new_regressors = ['is_weekend', 'rolling_std_mean_3', 'future_exceedence','hist_indoor', 'temp_gradient']

      X['y_data']=y
#      if corr_plot: 
#          import matplotlib.pyplot as plt
#          import seaborn as sns
#          plt.figure()
#          sns.heatmap(X.corr())
#          X = X.dropna()
#          y=X['y_data']
        
#          # Print correlation values to help hint at what regression parameters
#          # to choose
#          CorrMatrix=pd.DataFrame(X.corr()['y_data'])
#          print(CorrMatrix.sort_values(by=['y_data']))
#        
      X = X.drop(columns=['y_data'], axis=1)
    
      X=X.iloc[2:]
      y=y.iloc[2:]
      return X, y 
        
def split_train_test(X, y, pct_train=0.8, month_range=None, test_year=None):
    
    '''
    Establish train/test split | Default value should be 80% for train, 20% for test, 
    but should be considered against the type of variable we're training against

    
    '''
    percentTrain = pct_train
    
    if month_range: 
        month_low = month_range[0]
        month_high = month_range[1]
    else: 
        month_low = 6
        month_high = 10

    # Establish training data start and stop

    trainDateStart = X.index[0].date().strftime('%Y-%m-%d %X')
    trainDateStop = X.index[round(len(X)*percentTrain)].date().strftime('%Y-%m-%d %X')
    testDateStart = X.index[round(len(X)*percentTrain)+1].date().strftime('%Y-%m-%d %X')
    testDateStop = X.index[len(X)-1].date().strftime('%Y-%m-%d %X')
       
    # Establish training and testing datasets
    
    trainX = X[trainDateStart:trainDateStop].astype(float)
    trainY = y[trainDateStart:trainDateStop].astype(float)
    
    testX = X[testDateStart:testDateStop].astype(float)
    testY = y[testDateStart:testDateStop].astype(float)
    
    if test_year: 
        
        dr = list(range(X.index.year.min(), X.index.year.max()+1))
        # Establish training and testing datasets according to leaving year out 
        dr.remove(test_year)
        
        trainX = X[(X.index.year==dr[0]) |
                   (X.index.year==dr[1]) |
                   (X.index.year==dr[2]) |
                   (X.index.year==dr[3]) ].astype(float)
        
        trainY = y[(y.index.year==dr[0]) |
                   (y.index.year==dr[1]) |
                   (y.index.year==dr[2]) |
                   (y.index.year==dr[3]) ].astype(float)
        
        testX = X[X.index.year==test_year].astype(float)
        
        testY = y[y.index.year==test_year].astype(float)
    
    trainX=trainX[(trainX.index.month > month_low) & (trainX.index.month < month_high)]
    trainY=trainY[(trainY.index.month > month_low) & (trainY.index.month < month_high)]
    
    testX=testX[(testX.index.month > month_low) & (testX.index.month <month_high)]
    testY=testY[(testY.index.month > month_low) & (testY.index.month <month_high)]
    
    
    print('Train Start Date: ', trainX.index[0].date().strftime('%Y-%m-%d %X'))
    print('Train Stop Date:  ', trainX.index[-1].date().strftime('%Y-%m-%d %X'))
    print('Test Start Date:  ', testX.index[0].date().strftime('%Y-%m-%d %X'))
    print('Test Start Date:  ', testX.index[-1].date().strftime('%Y-%m-%d %X'))
    
    return trainX, trainY, testX, testY