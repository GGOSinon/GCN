import numpy as np
import random
from Net_data import Net, Connection, Node
import copy

def softmax(x):
    max_x = np.max(x)
    e_x = np.exp(x-max(x))
    return e_x / e_x.sum()

class SuperNeat:
    def __init__(self):
        self.Nets = []
        self.num_conn = {} # num_conn[num] = conn
        self.num_node = {} # num_node[num] = node
        self.max_num_node = 0
        self.max_num_conn = 0
    
    def run(self, input_size, output_size, trainloader, testloader, repeat_num = 100, net_num = 128, node_num = 100, node_size = 10, mutate_node_rate = 0.4, mutate_conn_rate = 0.5, crossover_rate = 0.4, mutate_node_del_rate = 0.7, mutate_conn_del_rate = 0.7):
        self.trainloader = trainloader
        self.testloader = testloader
        self.mutate_node_rate = mutate_node_rate
        self.mutate_conn_rate = mutate_conn_rate
        self.mutate_node_del_rate = mutate_node_del_rate
        self.mutate_conn_del_rate = mutate_conn_del_rate
        self.crossover_rate = crossover_rate
        self.net_num = net_num
        self.node_size = node_size

        self.num_node[0] = Node(input_size)
        self.num_node[1] = Node(output_size)
        net = Net()
        for i in range(net_num):
            #net = Net()
            #if i==0:net.connections[0]=Connection(0,1)
            #print(net.connections)
            self.Nets.append(copy.deepcopy(net))

        for net in self.Nets:
            #print("Start")
            #print(net.nodes, net.connections)
            net.add_node(0, input_size)
            net.add_node(1, output_size)
            net.add_connection(0, 0, 1)
            #print(net.nodes, net.connections)
        self.max_num_node = 2
        self.max_num_conn = 1
        step = 0
        acc = []
        while step < repeat_num:
            step += 1
            print("Step "+str(step)+" started")
            maxfitness = 0
            for net in self.Nets:
                self.evaluate(net)
                if maxfitness<net.fitness:
                   maxfitness = net.fitness
                   BestNet = net
            newNets = []
            p = np.zeros(net_num)
            for i in range(net_num): p[i] = self.Nets[i].fitness
            p = softmax(p)
            for i in range(1,net_num): p[i] += p[i-1]
            for net in self.Nets:
                newNets.append(self.make_child(p))
            self.Nets = newNets
            acc.append(int(BestNet.fitness*100))
        for i in len(acc):
            print("Step "+str(i)+": Accuracy is "+str(acc[i])+"%%")
        return BestNet

    def evaluate(self, net):
        print(net)
        net.train(self.trainloader)
        net.test(self.testloader)

    def make_child(self, p):
        #print(p)
        Nets = self.Nets
        net_num = self.net_num
        if random.random() < self.crossover_rate:
            r = random.random()
            for i in range(net_num):
                if p[i]>r:
                    pos = i
                    break
            net1 = Nets[pos]
            r = random.random()
            for i in range(net_num):
                if p[i+1]>r:
                    pos = i
                    break
            net2 = Nets[pos]
            child = self.crossover(net1, net2)
        else:
            r = random.random()
            for i in range(net_num):
                if p[i]>r:
                    pos = i
                    break
            child = copy.deepcopy(Nets[pos])
        
        child = self.mutate(child)
        return child

    def mutate(self, net):
        r = random.random()
        if r<self.mutate_conn_rate:
            net = self.mutate_add_connection(net)
        r = random.random()
        if r<self.mutate_node_rate:
            net = self.mutate_add_node(net)
        r = random.random()
        if r<self.mutate_conn_del_rate:
            net = self.mutate_del_connection(net)
        return net

    def crossover(self, net1, net2):
        if net2.fitness>net1.fitness:
            temp = net1
            net1 = net2
            net2 = temp
    
        num2 = {}
        child = copy.deepcopy(Net())
        for num in net1.nodes:
            node = net1.nodes[num]
            child.add_node(num, node.size)
        for num in net2.nodes:
            node = net2.nodes[num]
            child.add_node(num, node.size)
        for num in net2.connections:
            conn = net2.connections[num]
            num2[conn.num] = conn
        for num in net1.connections:
            conn = net1.connections[num]
            r = random.randrange(0,2)
            if conn.num in num2 and r==1 and num2[conn.num].enabled:
                conn2 = num2[conn.num]
                child.add_connection(conn2.num, conn2.s, conn2.e)
            else:
                child.add_connection(conn.num, conn.s, conn.e)
        return child
    
    def add_connection(self, net, s, e):
        conn = Connection(s, e)
        net.add_connection(self.max_num_conn, s, e)
        self.num_conn[self.max_num_conn] = conn
        self.max_num_conn += 1 
        return net       

    def find_with_index(self, dic, pos):
        for i, key in enumerate(dic):
            if i == pos: return dic[key]

    def mutate_add_node(self, net):
        r = random.randrange(0, self.node_size)
        size = 2**r
        connections = net.connections
        p = random.randrange(0,len(connections))
        conn = self.find_with_index(connections, p)
        node = net.add_node(self.max_num_node, size)
        net.disable(conn.num)
        s = conn.s
        e = conn.e
        self.add_connection(net, s, self.max_num_node)
        self.add_connection(net, self.max_num_node, e)
        self.max_num_node += 1
        return net

    def mutate_add_connection(self, net):
        connections = net.connections
        r = random.randrange(0, len(connections))
        conn = self.find_with_index(connections, r)
        s = conn.s
        e = conn.e
        if net.check(s,e):return net
        #while net.check_cycle(s, e)==0:
            #s = random.randrange(0, self.max_num_node)
            #e = random.randrange(0, self.max_num_node)
        net = self.add_connection(net, s, e)
        return net
    
    def mutate_del_connection(self, net):
        connections = net.connections
        p = random.randrange(0, len(connections))
        if p==0: return net
        conn = self.find_with_index(connections, p)
        net.disable(conn.num)
        return net
    
    def mutate_del_node(self, net):# Not implemented
        connections = net.connections
        p = random.randrange(0, len(net.nodes))
        node_num = self.find_with_index(net.nodes, p).key()
        for conn in connections:
            if conn.e == node_num:
                net.disable(conn)
                #add_connection(net, conn.s, conn.e)
            if conn.s == node_num:
                net.disable(conn)
        return net
     
