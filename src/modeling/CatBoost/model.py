import pickle 
import sys 
import catboost as cb
import pandas as pd
import numpy as np
sys.path.append('src')
from data.preprocess import read_processed_data

class CatBoostModel(cb.CatBoostClassifier):
    def __init__(self, config=None):
        super(CatBoostModel, self).__init__()
        config = {'eval_metric':"AUC", 'depth': 10, 'iterations': 200, 
                  'l2_leaf_reg': 4, 'learning_rate': 0.15, 'task_type':"GPU"}
            
        self.set_params(**config)
        
    def train(self, trainX, trainY):
        X = trainX
        y = trainY
        self.fit(X, y)
        
    def make_predictions(self, X, save_to):
        preds_class = self.predict(X)
        preds_proba = self.predict_proba(X)  
        preds_df = pd.DataFrame(data={'class':preds_class,
                                      'proba':preds_proba[:,1]}, 
                                      index = X.index)

        if save_to: 
            with open(save_to, 'wb') as outfile:
                pickle.dump(preds_df, outfile, pickle.HIGHEST_PROTOCOL)
                
        return preds_df
    
    def save(self, fname):
        with open(fname, 'wb') as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    def load(fname):
        with open(fname, 'rb') as infile:
            return pickle.load(infile)
