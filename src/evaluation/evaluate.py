import math
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import optimize
import sys
import click
sys.path.append('src')

def load_predictions(model = 'CatBoost', 
                     fname = 'data/output/',
                     thres = 0.8):
    
    with open(fname+model+"/preds.pkl", 'rb') as infile:
        df = pickle.load(infile)
    df['with_thres'] = np.where(df['proba']>=thres, 1, 0)
    return df

def load_data(model = 'CatBoost',
              fname = 'data/processed/'):

    with open(fname+model+"/train.pkl", 'rb') as infile:
        train = pickle.load(infile)   
    with open(fname+model+"/test.pkl", 'rb') as infile:
        test = pickle.load(infile) 
    
    trainY = train['trainY']
    train = train.drop('trainY', axis=1)
    trainX = train 
    testY = test['testY']
    test = test.drop('testY', axis=1)
    testX = test
    return trainX, trainY, testX, testY
    

def plot_predicted_vs_actual(model, predsData, testData, fname=None):
    import matplotlib.dates as mdates

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), sharey=True, dpi=120)
    font = "Times New Roman"    
    #set ticks every week
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator())
    #set major ticks format
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    
    #set ticks every week
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator())
    #set major ticks format
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax2.yaxis.major.formatter._useMathText = True

    ax1.plot(predsData,'b.')
    ax1.set_title('Predicted Hot Days',fontname=font,fontweight="heavy")
    ax1.set_ylabel('Probability',fontname=font, fontsize = 12)

    ax2.plot(testData,'r.')
    ax2.set_title('Actual Hot Days',fontname=font,fontweight="bold")
    plt.subplots_adjust(wspace=0.04, hspace=0)

    fig.autofmt_xdate()
    fig.patch.set_facecolor('white')
    if fname:
        fname = fname 
    else: 
        fname = 'figures/predicted_vs_actual_'+model+'thres80'+'.pdf'
    
    for ax in [ax1,ax2]:
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname(font) for label in labels]
    
    plt.savefig(fname,dpi=300)

    plt.show();


def plot_cumulative_distr(preds_df):
    '''
    Takes the probabilities of a day being classified as hot, and calculates 
    the empirical cumulative distribution of probabilities.
    
    '''
    fig, ax = plt.subplots(1,1, figsize=(5,4), sharey=True, dpi=120)
    font = "Times New Roman"    

    # quantiles
    bins = np.linspace(0.5,0.99,100)
    xvals = []
    for i in range(len(bins)):
      min_lim = bins[i]
      xvals.append(preds_df[(preds_df['proba']>=min_lim)].count().values.item(0))
          
    def ecdf(data):
        """ Compute ECDF """
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n+1)/n
        return(x,y)
    
    bins, xvals = ecdf(xvals)

    ax.plot(bins, xvals,'b.')
    fig.show()
    
    # Cumulative distribution function 
    f2 = lambda x,mu,la: 0.5+0.5*scipy.special.erf((np.log(x)-mu)/((2**0.5)*la))

    mu,la = scipy.optimize.curve_fit(f2,np.array(bins),np.array(xvals))[0]
    
    x2=np.linspace(min(bins),max(bins),300)
    ax.plot(x2,f2(np.array(x2),mu,la))
    ax.set_ylabel('ECDF',fontname=font,fontweight="heavy",fontsize = 12)
    ax.set_xlabel('x',fontname=font,fontsize = 12)
    
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname(font) for label in labels]
    plt.show();
    
    return mu, la
    


def plot_prob_density(mu, la, predsData, testData):
    from scipy.stats import lognorm

    fig, axes = plt.subplots(1,1, figsize=(5,4), sharey=True, dpi=120)
    font = "Times New Roman"    

    f3 = lambda x,mu,la: (1/x*la*(2*math.pi)**0.5)*np.exp(-((np.log(x)-mu)**2)/(2*la**2))

    x2=np.linspace(0,10,300)

    axes.plot(x2,f3(x2,mu,la))
    ymin, ymax = axes.get_ylim()
    
    x_bounds = lognorm.interval(alpha=0.95, s=la, scale=np.exp(mu))
    x_bounds_std = lognorm.interval(alpha=0.68,s=la,scale=np.exp(mu))
    
    axes.axvline(x=testData.sum() ,color='red',linestyle=':')
    ymaxes= f3(np.asarray(x_bounds),mu,la)/ymax+0.01
    
    axes.axvline(x=x_bounds[0] ,color='blue',alpha=0.3,linestyle=':')
    axes.axvline(x=x_bounds[1] ,color='blue',alpha=0.3,linestyle=':')
    
    xfill =  np.linspace(x_bounds[0],x_bounds[1],100)
    xfill_std =  np.linspace(x_bounds_std[0],x_bounds_std[1],100)
    
    axes.fill_between(xfill,f3(xfill,mu,la),alpha=0.1,color='blue')
    axes.fill_between(xfill_std,f3(xfill_std,mu,la),alpha=0.1,color='blue')
    
    #axes.fill_between(xfill,)
    axes.text(x=testData.sum()+1,y=.03*ymax,s='Actual: '+str(int(testData.sum())),color='red')
    #axes.text(x=x_bounds[1]+1,y=ymax*.9,s='Upper 95%:',color='blue')
    #axes.text(x=x_bounds[1]+1,y=ymax*.82,s=str(round(x_bounds[1],1)),color='blue')
    #axes.text(x=x_bounds[0]-10,y=ymax*.9,s='Lower 95%:',color='blue')
    #axes.text(x=x_bounds[0]-10,y=ymax*.82,s=str(round(x_bounds[0],1)),color='blue')
    axes.set_xlabel('Number of days exceeding threshold',fontname=font,fontweight="heavy",fontsize = 12)
    axes.set_ylabel('Probability density function (-)',fontname=font,fontweight="heavy",fontsize = 12)
    axes.set_ylim(0,ymax)
    axes.set_xlim(0,10)
    
    labels = axes.get_xticklabels() + axes.get_yticklabels()
    [label.set_fontname(font) for label in labels]
    fig.show();
    
    
    print('**********************************')
    print('Expected number of days exceeding thermal comfort criteria: '+str(round(lognorm.mean(s=la,scale=np.exp(mu)),1))  + ' +/- ' + str(round(lognorm.std(s=la,scale=np.exp(mu)),1)))
    print('Most likely number of days exceeding thermal comfort criteria: '+str(round(np.exp(mu - la**2)))  + ' +/- ' + str(round(lognorm.std(s=la,scale=np.exp(mu)),1)))
    print('Predicted number of days exceeding thermal comfort criteria (deterministic): '+str(int(np.sum(predsData))))
    print('Actual number of days exceeding thermal comfort criteria: ' + str(int(testData.sum())))
    print('**********************************')
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
    acc_score = accuracy_score(predsData, testData)
    prec_score = precision_score(predsData, testData)
    rec_score = recall_score(predsData, testData)
    roc_auc_score = roc_auc_score(predsData, testData)
    print("Test Accuracy score: ", acc_score)
    print("Test Precision score: ", prec_score)
    print("Test Recall score: ", rec_score)
    print("Test ROC AUC score: ", roc_auc_score)



def boxplot(preds_df, testData):
        # quantiles
    bins = np.linspace(0.5,0.99,100)
    xvals = []
    for i in range(len(bins)):
      min_lim = bins[i]
      xvals.append(preds_df[(preds_df['proba']>=min_lim)].count().values.item(0))
          
    def ecdf(data):
        """ Compute ECDF """
        x = np.sort(data)
        n = x.size
        y = np.arange(1, n+1)/n
        return(x,y)
    
    bins, xvals = ecdf(xvals)
    
    fig, axes = plt.subplots(1,1, figsize=(5,4), sharey=True, dpi=120)
    axes.axhline(y=testData.sum() ,color='blue',linestyle=':')
    vals = pd.DataFrame(data = bins)
    
    return vals.boxplot()

#@click.command()
#@click.option('--model', default = 'CatBoost', show_default=True)
#@click.option('--cutoff', default = 0.8, show_default=True)
def main(model, cutoff):
    model = 'CatBoost' 
    cutoff = 0.8
    trainX, trainY, testX, testY = load_data()
    preds_df = load_predictions(thres=cutoff)
    preds_class = preds_df['with_thres']
    
    # Resample to daily when using hourly training data 
    preds_class = pd.Series(data = preds_class, index = testX.index).resample('D').max()
    testY = testY.resample('D').max()
    preds_df = preds_df.resample('D').mean()
    preds_class = pd.Series(data = preds_class, index = testX.index).resample('D').max()
    testY = testY.resample('D').max()
    
    plot_predicted_vs_actual(model = model, predsData = preds_class, 
                             testData = testY)
    mu, la = plot_cumulative_distr(preds_df)
    plot_prob_density(mu, la, predsData = preds_class, testData = testY)
    boxplot(preds_df, testData = testY)

if __name__ == '__main__':
    main()