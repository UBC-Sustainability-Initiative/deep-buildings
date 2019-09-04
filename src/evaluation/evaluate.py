import math
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
import seaborn
sys.path.append('src')

def load_predictions(model = 'CatBoost', 
                     fname = '../../data/output/'):
    
    with open(fname+model+"/preds.pkl", 'rb') as infile:
        return pickle.load(infile)

def load_data(model = 'CatBoost',
              fname = '../../data/processed/CatBoost/'):

    with open(fname+model+"/train.pkl", 'rb') as infile:
        train = pickle.load(infile)   
    with open(fname+model+"/test.pkl", 'rb') as infile:
        test = pickle.load(infile) 
    
    
    
    
    
preds_df = load_predictions()
preds_class = preds_df['80_confident']

## Resample to daily when using hourly training data 
#preds_class = pd.Series(data = preds_class, index = testX.index).resample('D').max()
#testY = testY.resample('D').max()
#preds_df = preds_df.resample('D').mean()
#
#preds_class = pd.Series(data = preds_class, index = testX.index).resample('D').max()
#testY = testY.resample('D').max()

plt.plot(preds_class,'b.')
plt.title('Predicted Hot Days')
plt.show()
plt.plot(testY.values,'r.')
plt.title('Actual Hot Days')
plt.show()

bins = [0,0.02,0.04,0.06,0.08,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95,0.99]
xvals = []
lastVal = 0
for i in range(len(bins)):
  min_lim = bins[i]
  xvals.append(preds_df[(preds_df['proba']>min_lim)].count().values.item(0))

plt.plot(xvals,1-np.array(bins),'b.')

# Cumulative distribution function 
#f = lambda x,la: ((2**0.5)/(la*(math.pi)**0.5))*np.exp(-(x**2)/(2*la**2))
f2 = lambda x,mu,la: 0.5+0.5*scipy.special.erf((np.log(x)-mu)/((2**0.5)*la))

mu,la = scipy.optimize.curve_fit(f2,np.array(xvals),1-np.array(bins))[0]

x2=np.linspace(0,90,300)
plt.plot(x2,f2(np.array(x2),mu,la))
plt.show()

#
f3 = lambda x,mu,la: (0.5/x*la*(2*math.pi)**0.5)*np.exp(-((np.log(x)-mu)**2)/(2*la**2))

from scipy.stats import lognorm