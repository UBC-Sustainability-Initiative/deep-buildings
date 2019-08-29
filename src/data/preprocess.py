import pandas as pd 



def read_interim_data(fname='data/interim/CIRS_data_joined_LSTM.pkl'):
    dframe = pd.read_pickle(fname)
    return dframe


def get_features(df: pd.DataFrame, predictors: list) -> pd.DataFrame:
    return df[predictors] # maybe change to extract up to last column

def get_target(df: pd.DataFrame, target: str) -> pd.DataFrame: 
    return df[target] # maybe change to extract last column


def preprocess_data(df: pd.DataFrame, regr_vars: list, target: str) -> pd.DataFrame:
    '''
    Takes merged dataframe, regression variables and target. 
    
    See this link for more details: "https://stackoverflow.com/questions/305330
    21/interpolate-or-extrapolate-/only-small-gaps-in-pandas-dataframe"

    Returns: 
        - X: interpolated where gap is under 6 hours in length 
        - y: target data 
    
    '''
    
    X = df[regr_vars]
    X['target_data'] = df[target]
    X = X.groupby(X.index).first() # Drop additional duplicates in X
    
    mask = X.copy()
    grp = ((mask.notnull() != mask.shift().notnull()).cumsum())
    grp['ones'] = 1
    for col in mask.columns:
        mask[col] = (grp.groupby(col)['ones'].transform('count') < 6) | X[col].notnull()
    X = X.interpolate().bfill()[mask]
    
    X = X.dropna(how="any")
    y = X['target_data']
    X = X.drop(columns=['target_data'])

    return X, y 

def read_processed_data(fname='data/processed/processed.pkl'):
    df = pd.read_pickle(fname)
    return df

def main(input_file: str, output_file: str, regr_vars: list, target: str):
    print('Preprocessing data')
    df = read_interim_data(input_file)
    df = preprocess_data(df, regr_vars, target)
    df.to_pickle(output_file)
    
if __name__ == '__main__':
    main()