import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import gumbel_utils as gu

class GraphNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, is_classifier):
        super(GraphNetwork, self).__init__()
        self.edge1 = torch.nn.Linear(2 * input_size, hidden_size)
        self.edge2edge = torch.nn.Linear(hidden_size, hidden_size)
        self.node2node = torch.nn.Linear(hidden_size, hidden_size)
        self.node2node2 = torch.nn.Linear(hidden_size, hidden_size)
        #self.output1 = torch.nn.Linear(hidden_size+input_size, hidden_size)
        #self.output2 = torch.nn.Linear(hidden_size, input_size)
        self.output = torch.nn.Linear(input_size+hidden_size, input_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)
        self.is_classifier = is_classifier
    def forward(self, x, adj):
        out = x
        innode = out.unsqueeze(1).repeat(1, adj.size()[1], 1, 1)
        outnode = innode.transpose(1, 2)
        node2edge = F.relu(self.edge1(torch.cat((innode,outnode), 3)))
        edge2edge = F.relu(self.edge2edge(node2edge))
        adjs = adj.view(adj.size()[0], adj.size()[1], adj.size()[2], 1)
        adjs = adjs.repeat(1, 1, 1, edge2edge.size()[3])

        edges = adjs * edge2edge

        out = torch.sum(edges, 1)
        out1 = F.relu(self.node2node(out))
        out2 = F.relu(self.node2node2(out1))
        out3 = torch.cat((x, out2), dim=-1)
        #out4 = F.relu(self.output1(out3))
        #out5 = F.relu(self.output2(out4))
        out5 = self.output(out3)
        out6 = out5+x
        return out6 if not self.is_classifier else self.logsoftmax(out5)


class SigmoidMatrix(nn.Module):
    def __init__(self, num_nodes):
        super(SigmoidMatrix, self).__init__()
        #self.matrix = Parameter(torch.full(size=(num_nodes, num_nodes), fill_value=0.5))
        self.matrix = Parameter(0.1*torch.randn(size=(num_nodes, num_nodes)))  # pre-sigmoid edge values
        
    def get_matrix(self, raw=False):
        return self.matrix if raw else torch.sigmoid(3*self.matrix)

    def get_matrix_hard(self):
        return (self.matrix > 0).float()

# credit to Zhang et al
class GumbelGenerator(nn.Module):
    def __init__(self, num_nodes, temp=10, temp_drop_frac=0.9):
        super(GumbelGenerator, self).__init__()
        self.matrix = Parameter(torch.rand(num_nodes, num_nodes, 2)) # unnormalized log probabilities for "egde" and "no edge"
        self.temperature = temp
        self.temp_drop_frac = temp_drop_frac
        
    def get_temperature(self):
        return self.temperature
    
    def drop_temperature(self):
        self.temperature = self.temperature * self.temp_drop_frac

    def get_matrix(self):
        # sampling
        self.logp = self.matrix.view(-1, 2)  # convert to 100x2 matrix instead of 10x10x2
        out = gu.gumbel_softmax_sample(self.logp, self.temperature)
        # convert to 10x10 (entries represent probability that there is an edge)
        out_matrix = out[:, 0].view(self.matrix.size()[0], self.matrix.size()[0])
        return out_matrix

    def get_matrix_hard(self):
        return (torch.min(self.matrix.data, 2)[1]).float()

    def print_logits(self):
        print(self.matrix.detach().cpu().numpy())