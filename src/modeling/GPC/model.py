import pickle
import torch
from gpytorch.models import AbstractVariationalGP
from gpytorch.variational import AdditiveGridInterpolationVariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from gpytorch.likelihoods import BernoulliLikelihood
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal

class GPClassificationModel(AbstractVariationalGP):
    def __init__(self, data_dim, grid_size=64, grid_bounds=([-1, 1],)):
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=grid_size, batch_size=data_dim)
        variational_strategy = AdditiveGridInterpolationVariationalStrategy(self,
                                                                            grid_size=grid_size,
                                                                            grid_bounds=grid_bounds,
                                                                            num_dim=data_dim,
                                                                            variational_distribution=variational_distribution)
        super(GPClassificationModel, self).__init__(variational_strategy)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel(nu=1.5))#RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_pred = MultivariateNormal(mean_x, covar_x)
        return latent_pred
    
    
    def make_predictions(self, X, save_to):
        #TODO ADAPT TO GPC
        
#        preds_class = self.predict(X)
#        preds_proba = self.predict_proba(X)  
#        preds_df = pd.DataFrame(data={'class':preds_class,
#                                      'proba':preds_proba[:,1]}, 
#                                      index = X.index)
#
#        if save_to: 
#            with open(save_to, 'wb') as outfile:
#                pickle.dump(preds_df, outfile, pickle.HIGHEST_PROTOCOL)
#                
        return 0
    
    def save(self, fname):
        torch.save(self.state_dict(), fname)
            
    def load(self, fname):
        state_dict = torch.load(fname)
        return self.load_state_dict(state_dict)
