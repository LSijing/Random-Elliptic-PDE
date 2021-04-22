# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 13:08:58 2019


num_basis= 20, layers 1+4+1, hidden size 50, activation tanh, lr 0.01(/4**i), random seed 1234

num_basis= 10, layers 1+4+1, hidden size 50, activation tanh, lr 0.01(/4**i), random seed 1234

num_basis=  5, layers 1+4+1, hidden size 50, activation tanh, lr 0.01(/4**i), random seed 1234




@author: li-sj13
"""



import numpy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import time
import scipy.io as sio
from scipy.sparse import csr_matrix


class local_POD_map(nn.Module):
    def __init__(self):
        super(local_POD_map,self).__init__()
        self.size = 50
        self.inp = nn.Linear(20,self.size)
        self.l1 = nn.Linear(self.size,self.size)
        self.l2 = nn.Linear(self.size,self.size)
        self.l3 = nn.Linear(self.size,self.size)
        self.l4 = nn.Linear(self.size,self.size)
        self.out = nn.Linear(self.size,10)
        self.acti = F.tanh
        
    def forward(self,x):
        y = self.inp(x)
        y = self.acti(self.l1(y))+y
        y = self.acti(self.l2(y))+y
        y = self.acti(self.l3(y))+y
        #y = self.acti(self.l4(y))+y

        y = self.out(y)
        return y
        

if __name__ == '__main__':
    start = time.clock()
    torch.manual_seed(1234)
    net = local_POD_map()
    net.double()
    
    sio.loadmat
    
    with h5py.File('Example4_18rvs2_local_data.mat', 'r') as f:
        num_basis = 10
        training_size = int(f['training_size'][0][0])
        Xdata = torch.from_numpy(f['Xinput'][0:training_size,:].transpose())
        Ydata = torch.from_numpy(f['Youtput'][0:training_size,0:num_basis].transpose())
        # data normalization
        Ave = Ydata.mean()
        Std = Ydata.std()
        Ydata = (Ydata - Ave) / Std
        
        # 
        zeta = f['Phi'][0:num_basis].transpose()
        zetator = torch.from_numpy(zeta)
        
    with h5py.File('revision1_Ex4_local_data.mat', 'r') as f:
        Xtest = torch.from_numpy(f['Xonline'][:].transpose())
        testing_size = Xtest.size(1)
        
    learning_rate = 0.01
    criterion = nn.MSELoss()
    training_time_array = numpy.zeros(0)
    training_loss_array = numpy.zeros(0)
    testing_loss_array = numpy.zeros(0)
    L2err_array = numpy.zeros(0)
    L2proj_array = numpy.zeros(0)
    Ypred_array = numpy.zeros([num_basis,testing_size*int(training_size/500)*10])
    
    for i in range(6):
        optimizer = torch.optim.Adam(net.parameters(),lr = learning_rate/(4**i))
        for j in range(training_size):
            x = Xdata[:,j]
            y = Ydata[:,j]
            yout = net(x)
            loss = ((y-yout)**2).mean()
            if (j % 500 ==1):
                print('%d,%d,%f'%(i,j,loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_time_array = numpy.concatenate([training_time_array,
                                                     numpy.array(i*training_size+j+1).reshape(1)])
            training_loss_array = numpy.concatenate([training_loss_array,
                                            ((net(Xdata.permute(1,0))*Std+Ave-(Ydata*Std+Ave).permute(1,0)) ** 2).mean(1).mean().detach().numpy().reshape(1)],0)


            if (j % 500 == 499):
                Ypred = (net(Xtest.permute(1,0)) *Std + Ave).permute(1,0).detach().numpy()
                Ypred_array[:,(i*int(training_size/500)+round(j/500)-1)*testing_size:(i*int(training_size/500)+round(j/500))*testing_size] = Ypred
                
                
    end = time.clock()
        
    sio.savemat('revision1_Ex4_onlineoutput10.mat', {'t4_train':end-start,'training_time_array':training_time_array,'training_loss_array':training_loss_array,'Ypred_array':Ypred_array})
    


    
