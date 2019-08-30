import click
import sys
sys.path.append('src')
from CatBoost.model import CatBoostModel 
from CatBoost.get_data import generate_data, split_train_test
from data.preprocess import read_processed_data




@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.argument('scenario', default = 2, show_default=True)
@click.option('--regr_vars', default = ['solar_radiation','temp','wind_dir',
                                        'hum_ratio','windspeed','weekday',
                                        'week'], show_default=True)
@click.option('--multiplier', default = [-777,-777,-999,-999,-999,-999,-999],
              show_default = True)
@click.option('--baseline', default = [0,0,0,0,0,0,0], show_default = True)

@click.option('--look_back', default = [2,3,1,2,1,0,0], show_default = True)

@click.option('--look_ahead', default = [0,0,0,0,0,0,0], show_default = True)

@click.option('--corr_plot', default = False, show_default = True)

@click.option('--test_year', default = 2017, show_default=True)

def main(input_file, output_file, scenario, regr_vars, multiplier, baseline,
         look_back, look_ahead, corr_plot, test_year):
    
    df = read_processed_data(input_file)
    
    X, y= generate_data(df, freq='D', scenario=scenario, regr_vars = regr_vars,
                        multiplier = multiplier, baseline= baseline, 
                        look_back = look_back, look_ahead = look_ahead, 
                        corr_plot= corr_plot)
    
    trainX, trainY, testX, testY = split_train_test(X, y, test_year=test_year)
    print("Generated data for CatBoost")
    print('Training model...')
    model = CatBoostModel()
    model.train(trainX, trainY)
    model.save(output_file)

if __name__ == '__main__':
    main()