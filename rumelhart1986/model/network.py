"""Neural network architecture for the Rumelhart 1986 reproduction."""

# Implement the network architecture here.
import torch
import torch.nn as nn


class TreeNet(nn.Module):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.c1 = nn.Parameter(torch.empty(24, 6))
        self.c2 = nn.Parameter(torch.empty(12, 6))
        self.w1 = nn.Parameter(torch.empty(12, 6))
        self.w2 = nn.Parameter(torch.empty(6, 24))
        
        self.b_c1 = nn.Parameter(torch.zeros(6))
        self.b_c2 = nn.Parameter(torch.zeros(6))
        self.b_w1 = nn.Parameter(torch.zeros(6))
        self.b_w2 = nn.Parameter(torch.zeros(24))
        
        self.__init_weight()
        
    def __init_weight(self):
        for param in [self.c1, self.c2, self.w1, self.w2]:
            nn.init.uniform_(param, a=-0.3, b=0.3)
            
    def forward(self, person1, relationship):
        
        prep= torch.sigmoid(torch.matmul(person1, self.c1) + self.b_c1)
        relrep= torch.sigmoid(torch.matmul(relationship, self.c2) + self.b_c2)
        
        cont= torch.concat([prep, relrep], dim=1)
        
        hiden1 = torch.sigmoid(torch.matmul(cont, self.w1) + self.b_w1)
        output = torch.sigmoid(torch.matmul(hiden1, self.w2) + self.b_w2)
        
        return output
    

def encode(person_indx, relation_indx, n_perion=24, n_relation=12):
    
    person_onehot= torch.zeros(n_perion, dtype=torch.float32)
    person_onehot[person_indx]= 0.1
    rel_onehot= torch.zeros(n_relation, dtype= torch.float32)
    rel_onehot[relation_indx]= 0.1
    return person_onehot.unsqueeze(0), rel_onehot.unsqueeze(0)


