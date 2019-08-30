import click
import sys
from LSTM.model import Model 
from LSTM.get_data import generate_data, split_train_test
from data.preprocess import read_processed_data
sys.path.append('src')

@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--hist_keys', default = ['hum_ratio','hours','solar_radiation',
                                        'temp','wind_dir','windspeed',
                                        'L3S_Office_1'], show_default=True)

@click.option('--regr_vars', default = ['hum_ratio','hours','solar_radiation',
                                        'temp','wind_dir','windspeed',
                                        'L3S_Office_1'], show_default=True)
@click.option('--test_year', default = 2017, show_default=True)
def main(input_file, output_file, hist_keys, regr_vars, test_year, n_epochs):
    df = read_processed_data(input_file)
    X, y= generate_data(df, freq='D', regr_vars=regr_vars, 
                        hist_keys = hist_keys, hist_steps=2)
    trainX, trainY, testX, testY = split_train_test(X, y, test_year=test_year)
    print("Generated data for LSTM")
    print('Training model...')
    model = Model(dict(features=5, forecast_horizon=1)).cuda()
    model.batch_train(trainX, trainY, n_epochs=n_epochs, lr=0.0005)
    model.save(output_file)

if __name__ == '__main__':
    main()