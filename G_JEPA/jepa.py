import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from G_JEPA.utils.logger import logger
from G_JEPA.utils.memory import Memory


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
    


class ResidualLinearBlock(nn.Module):
    """
    Residual MLP block:
      x -> Linear(dim->mlp_dim) -> (LN) -> ReLU -> Linear(mlp_dim->dim) -> (LN)
      -> +skip -> ReLU
    """
    def __init__(self, dim: int, mlp_dim: int = None, use_layernorm: bool = True):
        super().__init__()
        mlp_dim = dim if mlp_dim is None else mlp_dim

        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(mlp_dim)
            self.norm2 = nn.LayerNorm(dim)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.fc1(x)
        if self.use_layernorm:
            out = self.norm1(out)
        out = self.act(out)

        out = self.fc2(out)
        if self.use_layernorm:
            out = self.norm2(out)

        out = out + residual
        out = self.act(out)
        return out


class LinearResNetPredictor(nn.Module):
    """
    Drop-in "Predictor" replacement but deeper (num_blocks controls depth).

    Flow:
      x -> stem (in_dim->hidden_dim) -> [ResidualLinearBlock]*num_blocks
        -> head (hidden_dim->out_dim)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_blocks: int = 4,
        mlp_dim: int = None,          # internal expansion dim inside each block
        use_layernorm: bool = True,
    ):
        super().__init__()
        mlp_dim = hidden_dim if mlp_dim is None else mlp_dim

        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            nn.ReLU(inplace=True),
        )

        self.blocks = nn.Sequential(
            *[ResidualLinearBlock(hidden_dim, mlp_dim, use_layernorm) for _ in range(num_blocks)]
        )

        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x



class JEPA:
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize encoder and predictor
        self.encoder = Encoder(config['encoder_input_dim'], config['encoder_hidden_dim'], config['encoder_output_dim']).to(self.device)
        self.predictor = Predictor(config['pred_input_dim'], config['pred_hidden_dim'], config['pred_output_dim']).to(self.device)
        self.predictor = LinearResNetPredictor(input_dim=config['pred_input_dim'], 
                                               hidden_dim=config['pred_hidden_dim'], 
                                               output_dim=config['pred_output_dim'], 
                                               num_blocks=config['pred_num_blocks'], 
                                               mlp_dim=config['pred_mlp_dim'])
        self.target_encoder = Encoder(config['encoder_input_dim'], config['encoder_hidden_dim'], config['encoder_output_dim']).to(self.device)

        self.memory = Memory()

        # Optimizer
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.predictor.parameters()), lr=config['learning_rate'])

        # Loss function
        self.criterion = nn.MSELoss()

    def count_trainable_params(self, model):
        """
        Returns total number of trainable parameters.
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    def print_jepa_params(self):
        encoder_params = self.count_trainable_params(self.encoder)
        predictor_params = self.count_trainable_params(self.predictor)
        target_encoder_params = self.count_trainable_params(self.target_encoder)

        total_params = encoder_params + predictor_params + target_encoder_params

        logger.info("========== JEPA Trainable Parameters ==========")
        logger.info(f"Encoder        : {encoder_params / 1e6:.3f} M")
        logger.info(f"Predictor      : {predictor_params / 1e6:.3f} M")
        logger.info(f"Target Encoder : {target_encoder_params / 1e6:.3f} M")
        logger.info("-----------------------------------------------")
        logger.info(f"TOTAL          : {total_params / 1e6:.3f} M")
        logger.info("===============================================")

    
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


    def learn(self):
        obs, actions, next_obs = self.memory.sample_memory()

        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(self.device)
        # we store actions as int
        actions = F.one_hot(torch.tensor(actions, dtype=torch.long), num_classes=self.config['action_dim']).float().to(self.device)

        with torch.no_grad():
            next_obs_features = self.target_encoder(next_obs)

        input_feature = torch.cat([self.encoder(obs), actions], dim=-1)

        pred_next_obs = self.predictor(input_feature)
        loss = self.criterion(pred_next_obs, next_obs_features)

        return loss
    

    def soft_update(self):
        for param, target_param in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target_param.data = self.config['tau'] * target_param.data + (1 - self.config['tau']) * param.data


    def save_checkpoint(self, filename="jepa.pth"):
        """Save model + optimizer for exact training resumption."""
        os.makedirs("models", exist_ok=True)
        checkpoint = {
            'encoder_state_dict': self.encoder.state_dict(),
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        save_path = os.path.join("models", filename)
        torch.save(checkpoint, save_path)
        logger.info(f"[SAVE] Checkpoint saved to {save_path}")


    def load_checkpoint(self, folder_name=None, filename="actor_critic_checkpoint.pth", load_optimizer=True):
        """Load model + optimizer state."""
        if folder_name is not None:
            file_path = os.path.join(folder_name, filename)
        else:
            file_path = os.path.join("models", filename)
        if not os.path.exists(file_path):
            logger.info(f"[LOAD] No checkpoint found at {file_path}")
            return False

        checkpoint = torch.load(file_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"[LOAD] Checkpoint loaded from {file_path}")
        return True




