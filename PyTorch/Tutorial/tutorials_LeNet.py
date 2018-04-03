import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

#Define the network
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)     #input 1 x 32x32
        self.conv2 = nn.Conv2d(6, 16, 5)    #class, def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True):
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)   #def __init__(self, in_features, out_features, bias=True):
        self.fc2 = nn.Linear(120, 84)           #y=Ax+b
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  #torch.Size([1,16,5,5])
        x = x.view(-1, self.num_flat_features(x)) # the size -1 is inferred from other dimensions
            #until now x: torch.Size([1,16*5*5])
        x = F.relu(self.fc1(x)) #torch.Size
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension    #torch.Size([16,5,5])
        num_features = 1
        for s in size:
            num_features *= s       # = 1*16*5*5
        return num_features

#test the network
net = Net()
print(net)

#learnable parameters
params = list(net.parameters())
#print(len(params))
#print(params[0].size())  # conv1's .weight



#Input
input = Variable(torch.randn(1,1,32,32))
out = net(input)        #net有参数？？？
#print(out)

net.zero_grad()
out.backward(torch.randn(1,10))

#input.unsqueeze()

nn.MSELoss

torch.arange

output = net(input)
target = Variable(torch.arange(1, 11))  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)
print(loss.grad_fn)  # MSELoss

params.data.sub_()



