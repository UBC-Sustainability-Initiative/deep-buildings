import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import pickle
import time
import numpy as np
import os
sys.path.append('src')


def create_dataset(trainX, trainY, look_back, forecast_horizon=1, batch_size=1):
    
    batch_x, batch_y, batch_z = [], [], []
    for i in range(0, len(trainX)-look_back-forecast_horizon-batch_size+1, batch_size):
        for n in range(batch_size):            
            x = trainX[['next_hum_ratio_1','next_hours_1','next_solar_radiation_1','next_temp_1','L3S_Office_1']].values[i+n:(i + n + look_back), :]
            offset = x[0, 0]
            y = trainY.values[i + n + look_back:i + n + look_back + forecast_horizon]
            
            batch_x.append(np.array(x).reshape(look_back, -1))
            batch_y.append(np.array(y))
            batch_z.append(np.array(offset))
        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        #print(batch_y)
        batch_z = np.array(batch_z)
        batch_x[:, :, 0] -= batch_z.reshape(-1, 1)
        batch_y -= batch_z.reshape(-1, 1)
        yield batch_x, batch_y, batch_z
        batch_x, batch_y, batch_z = [], [], []
        
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.hidden_size = 128
        self.bi = 1
        self.lstm = nn.LSTM(config.get('features'),self.hidden_size,1,dropout=0.1,bidirectional=bool(self.bi-1),batch_first=True)
        self.lstm2 = nn.LSTM(self.hidden_size,self.hidden_size // 4,1,dropout=0.1,bidirectional=bool(self.bi-1),batch_first=True)
        self.dense = nn.Linear(self.hidden_size // 4, config.get('forecast_horizon'))
        self.loss_fn = nn.MSELoss()
        
    def forward(self, x, batch_size=100):
        hidden = self.init_hidden(batch_size)
        output, _ = self.lstm(x, hidden)
        output = F.dropout(output, p=0.5, training=True)
        state = self.init_hidden2(batch_size)
        output, state = self.lstm2(output, state)
        output = F.dropout(output, p=0.5, training=True)
        output = self.dense(output[:,1,:])
        #output = self.dense(state[0].squeeze(0))
        
        return output
        
    def init_hidden(self, batch_size):
        h0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size).cuda())
        c0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size).cuda())
        return h0, c0
    
    def init_hidden2(self, batch_size):
        h0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size//4).cuda())
        c0 = Variable(torch.zeros(self.bi, batch_size, self.hidden_size//4).cuda())
        return h0, c0
    
    def loss(self, pred, truth):
        return self.loss_fn(pred, truth)
    
    def batch_train(self, trainX, trainY, n_epochs, lr, look_back=24, forecast_horizon=1,
              batch_size = 24):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()
        train_true_y = []
        train_pred_y = []
        for epoch in range(n_epochs):
            ep_loss = []
            for i, batch in enumerate(create_dataset(trainX, trainY, look_back=look_back, forecast_horizon=forecast_horizon, batch_size=batch_size)):
                print("[{}{}] Epoch {}: loss={:0.4f}".format("-"*(20*i//(len(trainX)//batch_size)), " "*(20-(20*i//(len(trainX)//batch_size))),epoch, np.mean(ep_loss)), end="\r")
                try:
                    batch = [torch.Tensor(x) for x in batch]
                except:
                    break
        
                out = self.forward(batch[0].float().cuda(), batch_size)
                loss = self.loss(out, batch[1].float().cuda())
                if epoch == n_epochs - 1:
                    train_true_y.append((batch[1] + batch[2]).detach().numpy().reshape(-1))
                    train_pred_y.append((out.cpu() + batch[2]).detach().numpy().reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ep_loss.append(loss.item())
            print()
                    
    def save(self, fname):
        fname = time.strftime("%Y%m%d-%H%M%S") + '_'+fname
        with open(fname, 'wb') as outfile:
            pickle.dump(model.state_dict(), outfile, pickle.HIGHEST_PROTOCOL)
#        
#    def load(self, fname):
#        with open(fname, 'rb') as infile:
#            self.model = pickle.load(infile)
