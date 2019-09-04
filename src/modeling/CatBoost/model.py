import pickle 
import sys 
import catboost as cb
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
        
#    def predict(self, X):
#        y_pred = self.predict(X)
#        return y_pred

    def save(self, fname):
        with open(fname, 'wb') as outfile:
            pickle.dump(self, outfile, pickle.HIGHEST_PROTOCOL)

    def load(self, fname):
        with open(fname, 'rb') as infile:
            return pickle.load(infile)
