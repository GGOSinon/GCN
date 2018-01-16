from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
#from Net_data import Net, Connection, Node
from NEAT import SuperNeat

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='./data',train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False,num_workers=2)
classes = ('plane', 'car','bird','cat','deer','dog','frog','horse','ship','truck')


NEAT = SuperNeat()
Net = NEAT.run(net_num = 10, repeat_num = 10, node_size = 5, input_size = 3*32*32, output_size = 10, trainloader = trainloader, testloader = testloader)
print(Net)
'''
node = [Node(3*32*32), Node(len(classes), F.sigmoid), Node(10)]
connection = [Connection(0,2), Connection(2,1)]
net = Net(node, connection)
print(net)

for epoch in range(2):  # loop over the dataset multiple times
    net.train(trainloader, 1)
    net.test(testloader)
'''
