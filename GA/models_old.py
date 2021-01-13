import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import *

class GumbelGraphNetworkClf(nn.Module):
    def __init__(self, input_size, hidden_size = 256):
        super(GumbelGraphNetworkClf, self).__init__()
        self.edge1 = torch.nn.Linear(2 * input_size, hidden_size)
        self.edge2edge = torch.nn.Linear(hidden_size, hidden_size)
        self.node2node = torch.nn.Linear(hidden_size, hidden_size)
        self.node2node2 = torch.nn.Linear(hidden_size, hidden_size)
        self.output = torch.nn.Linear(hidden_size, input_size)
        self.output1 = torch.nn.Linear(input_size+hidden_size, hidden_size)
        self.logsoftmax = nn.LogSoftmax(dim=2)
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
        out4 = self.output1(out3)
        out5 = self.output(out4)
        if not CONTINUOUS:
            out5 = self.logsoftmax(out5)

        return out5