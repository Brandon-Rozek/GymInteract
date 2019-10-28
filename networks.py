
import torch
import torch.nn as nn
import torch.nn.functional as F
import rltorch.network as rn

class Value(nn.Module):
  def __init__(self, state_size, action_size):
    super(Value, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    
    self.conv1 = nn.Conv2d(4, 32, kernel_size = (8, 8), stride = (4, 4))
    self.conv2 = nn.Conv2d(32, 64, kernel_size = (4, 4), stride = (2, 2))    
    self.conv3 = nn.Conv2d(64, 64, kernel_size = (3, 3), stride = (1, 1))
    
    self.fc1 = nn.Linear(3136, 512)
    self.fc1_norm = nn.LayerNorm(512)

    self.value_fc = rn.NoisyLinear(512, 512)
    self.value_fc_norm = nn.LayerNorm(512)
    self.value = nn.Linear(512, 1)
    
    self.advantage_fc = rn.NoisyLinear(512, 512)
    self.advantage_fc_norm = nn.LayerNorm(512)
    self.advantage = nn.Linear(512, action_size)

  
  def forward(self, x):
    x = x.float() / 256
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    
    # Makes batch_size dimension again
    x = x.view(-1, 3136)
    x = F.relu(self.fc1_norm(self.fc1(x)))
    
    state_value = F.relu(self.value_fc_norm(self.value_fc(x)))
    state_value = self.value(state_value)
    
    advantage = F.relu(self.advantage_fc_norm(self.advantage_fc(x)))
    advantage = self.advantage(advantage)
    
    x = state_value + advantage - advantage.mean()
    
    # For debugging purposes...
    if torch.isnan(x).any().item():
      print("WARNING NAN IN MODEL DETECTED")
    
    return x
