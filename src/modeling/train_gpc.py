import click
import sys
import gpytorch
import torch
sys.path.append('src')
from GPC.model import GPClassificationModel 
from GPC.get_data import generate_data, split_train_test
from data.preprocess import read_processed_data


@click.command()
@click.argument('input_file', type=click.Path(exists=True, readable=True, dir_okay=False))
@click.argument('output_file', type=click.Path(writable=True, dir_okay=False))
@click.option('--iterations', default = 500, show_default=True)
@click.option('--scenario', default = 2, show_default=True)
@click.option('--regr_vars', default = ['solar_radiation','temp','wind_dir',
                                        'hum_ratio','windspeed','weekday',
                                        'week'], show_default=True)

@click.option('--multiplier', default = [-777,-777,-999,-999,-999,-999,-999],
              show_default = True)
@click.option('--baseline', default = [0,0,0,0,0,0,0], show_default = True)

@click.option('--look_back', default = [2,3,1,2,1,0,0], show_default = True)

@click.option('--look_ahead', default = [0,0,0,0,0,0,0], show_default = True)

@click.option('--test_year', default = 2017, show_default=True)

def main(input_file, output_file, iterations, scenario, regr_vars, multiplier, baseline,
         look_back, look_ahead, test_year):
    
    df = read_processed_data(input_file)
    
    X, y= generate_data(df, freq='D', scenario=scenario, regr_vars = regr_vars,
                        multiplier = multiplier, baseline= baseline, 
                        look_back = look_back, look_ahead = look_ahead)
    
    trainX, trainY, testX, testY = split_train_test(X, y, test_year=test_year,
                                                    save_data = True,
                                                    pname = 'data/processed/GPC/')
    print("Generated data for GPC")
    print('Training model...')
    
    x_data_dim = trainX.size(-1)
    model = GPClassificationModel(data_dim = x_data_dim)
    # Cuda the model and likelihood function
    model = GPClassificationModel(data_dim = x_data_dim).cuda()
    likelihood = gpytorch.likelihoods.BernoulliLikelihood().cuda()
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    # Use the Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # "Loss" for GPs - the marginal log likelihood
    # n_data refers to the amount of training data
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=trainY.numel())
    # Training function
    def train(num_iter=iterations):
        for i in range(num_iter):
            optimizer.zero_grad()
            output = model(trainX)
            loss = -mll(output, trainY)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, num_iter, loss.item()))
            optimizer.step()
    
    train()

    model.save(output_file)
    print("Saved model to: ", output_file)
    model.make_predictions(testX, testY, likelihood, 
                           save_to = 'data/output/GPC/preds.pkl')

if __name__ == '__main__':
    main()