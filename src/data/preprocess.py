import pandas as pd 
import click 
pd.options.mode.chained_assignment = None


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
    #y = X['target_data']
    #X = X.drop(columns=['target_data'])
    df = X
    
    return df 

def read_processed_data(fname='data/processed/processed.pkl'):
    df = pd.read_pickle(fname)
    return df


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--regr_vars', default = ['hum_ratio','hours','solar_radiation',
                                        'temp','wind_dir','windspeed',
                                        'L3S_Office_1'],
    show_default=True)

@click.option('--target', default = 'indoorTemp', show_default=True)
def main(input_file: str, output_file: str, regr_vars=None, target=None):
    print('Preprocessing data...')
    
#    if regr_vars:
#        regr_vars = regr_vars
#    else: 
#        regr_vars = ['hum_ratio','hours','solar_radiation','temp','wind_dir',
#                     'windspeed','L3S_Office_1']
#        print('Using variables: ', regr_vars)
#    
#    if target:
#        target = target
#    else:
#        target = 'indoorTemp'
#        print('Using target: ', target)
        
    df = read_interim_data(input_file)
    df = preprocess_data(df, regr_vars, target)
    df.to_pickle(output_file)
    
if __name__ == '__main__':
    main()