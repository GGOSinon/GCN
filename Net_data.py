from __future__ import print_function
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import copy 

class Connection():
    def __init__(self, s, e):
        self.s = s
        self.e = e
        self.enabled = True
        self.num = 0

class Node():
    def __init__(self, size=1, actF=F.relu):
        self.size = size
        self.actF = actF

class Net(nn.Module):

    def __init__(self, nodes={}, connections={}):
        super(Net, self).__init__()
        self.nodes = copy.deepcopy(nodes)
        self.connections = copy.deepcopy(connections)
        self.num_fc = 0
        #self.fitness = 0
        for i in range(len(connections)):
            conn = connections[i]
            s = conn.s
            e = conn.e
            self.add_connection(i, s, e)
           #self.fc0 = nn.Linear(node[s].size, node[e].size)
            #self.fc.append(nn.Linear(node[s].size, node[e].size))

    def forward(self, x0):
        connections = self.connections
        nodes = self.nodes
        c = np.zeros(len(connections), dtype=np.int)
        deg = np.zeros(len(nodes), dtype=np.int)
        num_node = {}
        for i,key in enumerate(nodes):
            num_node[key]=i
        #print(num_node)
        for key in connections:
           conn = connections[key]
           if conn.enabled==False: continue
           e = num_node[conn.e]
           deg[e]+=1
        x = {}
        x[0] = x0.view(-1, 3*32*32)
        #print(num_node)
        #for num in connections:
        #    conn = connections[num]
        #    print(conn.s, conn.e)
        
        while True:
            #print(deg) 
            #print(connections)
            for i,key in enumerate(connections):
                conn = connections[key]
                if conn.enabled == False: continue
                s = conn.s
                e = conn.e
                ns = num_node[s]
                ne = num_node[e]
                if c[i] == 0 and deg[ns]==0:
                   c[i] = 1
                   variable_name = 'self.fc'+str(key)
                   F = getattr(self, variable_name)
                   if s not in x:
                       deg[ne] = 0
                       continue
                   X = F(x[s])
                   if e in x: x[e] += X
                   else: x[e] = X
                   deg[ne]-=1
                   if deg[ne]==0: x[s]=nodes[s].actF(x[s])
            #print(deg)
            done = True
            for i in range(len(nodes)):
                if deg[i]>0: done = False
            if done: break
        return x[1] #always 0 is input, 1 is output
    
    def check(self, s, e):
        connections = self.connections
        nodes = self.nodes
        num_node = {}
        for i, key in enumerate(nodes):
            num_node[key]=i
        #print(s,e)
        #print(num_node)
        #for num in connections:
        #    conn = connections[num]
        #    print(conn.s, conn.e)
        ns = num_node[s]
        ne = num_node[e]
        if s==e: return True
        q = np.zeros(len(connections)+len(nodes))
        q[0] = e
        sz = 1
        c = np.zeros(len(nodes))
        c[ne] = 1
        while sz>0:
            if c[ns]==1: return True
            x = q[0]
            sz -= 1
            q[0] = q[sz]
            for key in connections:
                conn = connections[key]
                s = conn.s
                e = conn.e
                ns = num_node[s]
                ne = num_node[e]
                if s==x and c[ne]==0:
                    c[ne] = 1
                    q[sz] = e
                    sz += 1
        return False

    def disable(self, num):
        self.connections[num].enabled = False

    def add_connection(self, num=0, s=0, e=0, conn=None):
        #print(self.connections)
        if not conn:
            if num in self.connections: return
            conn1 = Connection(s, e)
            conn1.num = num
            self.connections[num] = copy.deepcopy(conn1)
            variable_name = 'self.fc'+str(num)
            #print(variable_name, s, e)
            #print(self.nodes)
            setattr(self, variable_name, nn.Linear(self.nodes[s].size, self.nodes[e].size))
        else:
            self.connections[conn.num] = copy.deepcopy(conn)
            variable_name = 'self.fc'+str(conn.num)
            #print(variable_name, s, e)
            #print(self.nodes)
            setattr(self, variable_name, nn.Linear(self.nodes[conn.s].size, self.nodes[conn.e].size))
    
   
    def add_node(self, num, size):
        node = Node(size)
        self.nodes[num] = copy.deepcopy(node)

    def train(self, trainloader, epoch = 0):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 200 == 199:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
                correct = 0
                total = 0
                #if i+1>=1000:break

    def test(self, testloader):
        total = 0
        correct = 0
        for i, data in enumerate(testloader, 0):
            #if i>500:break
            images, labels = data
            outputs = self(Variable(images))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        self.fitness = float(correct) / total
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
