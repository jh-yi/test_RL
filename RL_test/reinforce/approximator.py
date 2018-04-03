
#!/home/qiang/PythonEnv/venv/bin/python3.5
# -*- coding: utf-8 -*-
# function approximators of reinforcment learning

# Author: Qiang Ye
# Date: July 27, 2017

import numpy as np
import torch
from torch.autograd import Variable
import copy


class Approximator(torch.nn.Module):
    '''base class of different function approximator subclasses

    #Functions
        1. (deep)NN that takes as inputs the state vector and an action vector,
        and output a single number that gives Q(s,a)
        2. update parameters w-> change Q(s,a)
    '''
    def __init__(self, dim_input = 1, dim_output = 1, dim_hidden = 16):
        '''set neural network layer

        super(...).__init__()
        '''
        super(Approximator, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.dim_hidden = dim_hidden

        self.linear1 = torch.nn.Linear(self.dim_input, self.dim_hidden)
        self.linear2 = torch.nn.Linear(self.dim_hidden, self.dim_output)


        #self.model = torch.nn.Sequential(
        #    torch.nn.Linear(self.dim_input, self.dim_hidden),
        #    torch.nn.ReLU(),
        #    torch.nn.Linear(self.dim_hidden, self.dim_output)
        #)
        pass

    def __call__(self, x):
        '''similar to predict function

        #Return an output given input.
            output is ndarray, consists of list([Q(s0)(a_0), Q(s0)(a_1),....]), describe all Q of all actions in state s0

        '''
        x=self._prepare_data(x)
        pred = self._forward(x)
        return pred.data.numpy()
        #raise NotImplementedError
    
    def _prepare_data(self, x, requires_grad = True):
        ''' change data x to the inquired size of Variabe: Torch input of NN:

        #Arguments
            x:raw data
            requires_grad: Boolean indicating whether the Variable has been
                created by a subgraph containing any Variable, that requires it.
                or: only = False when x is output


        #Return
            torch.FloatTensor of size[in_features x out_features]
        '''
        if isinstance(x, np.ndarray):
            x = Variable(torch.from_numpy(x), requires_grad = requires_grad)
        if isinstance(x, int):
            x = Variable(torch.Tensor([[x]]), requires_grad = requires_grad)
        x = x.float()   # 从from_numpy()转换过来的数据是DoubleTensor形式
        if x.data.dim() == 1:
            x = x.unsqueeze(0)
        return x

    def _forward(self, x):
        #for prediction Q(x) w.r.t. state x
        h_relu = self.linear1(x).clamp(min=0)
        y_pred = self.linear2(h_relu)
        return y_pred
        #raise NotImplementedError

    def fit(self, x, 
                  y, 
                  criterion=None, 
                  optimizer=None, 
                  epochs=1,
                  learning_rate=1e-4):
        #for NN training
        if criterion is None:
            criterion = torch.nn.MSELoss(size_average = False)
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        if epochs < 1:
            epochs = 1

        x = self._prepare_data(x)
        y = self._prepare_data(y, False)

        for t in range(epochs):
            y_pred = self._forward(x)

            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return loss

    def clone(self):
        '''返回当前模型的深度拷贝对象, DQN need 2 Q Network

        return
            an object of class Aprroximator , copy.deepcopy(self), complete copy & independent
        '''
        return copy.deepcopy(self)


def test():
    N, D_in, H, D_out = 64, 100, 50, 1
    x = Variable(torch.randn(N, D_in))
    y = Variable(torch.randn(N, D_out), requires_grad = False)

    model = Approximator(D_in, D_out, H)

    model.fit(x, y, epochs=1000)
    print('x[2]:{0}'.format(x[2]))
    #y_pred = model.__call__(x[2])
    y_pred = model(x[2])
    #y_pred = model.predict(x[2])
    print('y[2]:{0}'.format(y[2]))
    print('y_pred：{0}'.format(y_pred))
    new_model = model.clone()
    #new_pred = new_model.predict(x[2])
    new_pred = new_model.__call__(x[2])
    print('new_pred:{0}'.format(new_pred))
    print(model is new_model)


if __name__ == "__main__":
    test()
