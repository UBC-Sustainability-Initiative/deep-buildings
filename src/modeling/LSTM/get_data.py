from datetime import timedelta
import numpy as np
import pandas as pd

def generate_data(df, freq: str, regr_vars = None, hist_keys = None, 
                  hist_steps = None ):
    '''
    
    freq: either 'D' or 'H'
    

    
    '''
    X = df
    y = df['target_data']
    X = X.drop(columns=['target_data'])
    
    if regr_vars:
        regression_variables = regr_vars
        
    else: 
        regression_variables = list(X.columns)


    if hist_keys: 
        X_historyKeys = hist_keys
    else: 
        X_historyKeys = list(X.columns)
        
    if hist_steps:
        if type(hist_steps) == int: 
            X_NhistorySteps = np.ones(len(X_historyKeys), dtype=int)*hist_steps
        elif type(hist_steps) == list:
            X_NhistorySteps = hist_steps

    else: 
        X_NhistorySteps = np.zeros(len(X_historyKeys))
        
    regression_variables_handle= np.ones(len(regression_variables))
    # 0 - use actual (assumes value does not change daily)
    # 1 - use daily mean value
    # 2 - use daily mean, max, and min
    
    X = X[regr_vars]
    
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
      
#TODO    #X_NhistorySteps=[2,2,2,2,2,2,2] # 3 or above is broken!

    # 1 = Assign rolling average 
    # 2 = Assign only value of last time step
    # 3 = Assign value of last time step + 1st-order derivative of last time step
    # 4 = Assign value of last time step + 1st-order derivative of last time step + 2nd-order derivative of last time step
    
    X_dropCurrent=np.zeros(len(X_historyKeys))
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