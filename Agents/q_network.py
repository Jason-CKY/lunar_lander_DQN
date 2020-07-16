import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Linear Neural Net as a function approximator for state action values
    """ 
    def __init__(self, network_config):
        '''
        Assume network_config:dictionary contains:
            state_dim: int
            hidden_dim: int
            num_actions: int
        '''
        input_dim = network_config['state_dim']
        hidden_dim = network_config['hidden_dim']
        output_dim = network_config['num_actions']
        super(DQN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.main(x)
