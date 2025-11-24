import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from G_JEPA.utils.logger import logger



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Predictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    


class JEPA:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize encoder and predictor
        self.encoder = Encoder(config['encoder_input_dim'], config['encoder_hidden_dim'], config['encoder_output_dim']).to(self.device)
        self.predictor = Predictor(config['pred_input_dim'], config['pred_hidden_dim'], config['pred_output_dim']).to(self.device)
        self.target_encoder = Encoder(config['encoder_input_dim'], config['encoder_hidden_dim'], config['encoder_output_dim']).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=config['learning_rate'])

        # Loss function
        self.criterion = nn.MSELoss()

        
    
    def intrinsic_reward(self, obs, action, next_obs):
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            target_features = self.target_encoder(next_obs)

        action_one_hot = F.one_hot(torch.tensor(action, dtype=torch.long), num_classes=self.config['action_dim']).float().to(self.device)

        with torch.no_grad():
            pred_input = torch.cat([self.encoder(obs), action_one_hot], dim=-1)
            predicted_next_obs = self.predictor(pred_input)
        reward = self.criterion(predicted_next_obs, target_features)
        return reward.item()
