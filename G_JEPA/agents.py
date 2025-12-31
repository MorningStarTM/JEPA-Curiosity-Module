import torch 
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import grid2op
from G_JEPA.utils.converter import ActionConverter
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
import os
from G_JEPA.utils.logger import logger
import torch.optim as optim
from collections import defaultdict




class ResidualLinearBlock(nn.Module):
    def __init__(self, dim, hidden_dim=None, use_layernorm=True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm1 = nn.LayerNorm(hidden_dim)  # ← fix here
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



class LinearResNet(nn.Module):
    def __init__(self, in_dim=493, hidden_dim=512, num_blocks=3, out_dim=256):
        super().__init__()

        # stem: project input to hidden_dim
        self.stem = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # stack of residual MLP blocks
        self.blocks = nn.Sequential(
            *[ResidualLinearBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )

        # head: project to desired output dim
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x




class ActorCriticUP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.NEG_INF = -1e9

        self.affine = LinearResNet(
                in_dim=self.config.get('input_dim', 192),
                hidden_dim=self.config.get('hidden_dim', 512),
                num_blocks=self.config.get('num_blocks', 12),  
                out_dim=256,
            )


        self.action_layer = nn.Linear(256, self.config.get('action_dim', 58))
        self.value_layer  = nn.Linear(256, 1)

        self.logprobs, self.state_values, self.rewards = [], [], []

        # Optional: safer initializations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    @torch.no_grad()
    def _sanitize(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        x.clamp_(-1e6, 1e6)
        return x


    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ---------- ADD THIS ----------
    def _human_readable(self, n: int) -> str:
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.2f}K"
        else:
            return str(n)

    def param_counts(model: nn.Module):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable = total - trainable

        logger.info(f"Total params:        {total/1e6:.3f} M")
        logger.info(f"Trainable params:    {trainable/1e6:.3f} M")
        logger.info(f"Non-trainable params:{non_trainable/1e6:.3f} M")
    
    def forward(self, state_np):
        x = torch.from_numpy(state_np).float().to(self.value_layer.weight.device)
        x = self._sanitize(x)

        if x.dim() == 1:                      # ensure [B, F]
            x = x.unsqueeze(0)

        h = self.affine(x)                       # includes LayerNorm + ReLU
        h = torch.nan_to_num(h)                  # belt & suspenders

        logits = self.action_layer(h)
        logits = torch.nan_to_num(logits)        # if any NaN slipped through
        logits = logits - logits.max(dim=-1, keepdim=True).values           # stable softmax
        probs  = torch.softmax(logits, dim=-1)

        # final guard
        if not torch.isfinite(probs).all():
            # Zero-out non-finites and renormalize as an emergency fallback
            probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
            s = probs.sum()
            probs = (probs + 1e-12) / (s + 1e-12)

        dist   = Categorical(probs=probs)
        action = dist.sample()

        self.logprobs.append(dist.log_prob(action).squeeze(-1))
        self.state_values.append(self.value_layer(h).squeeze(-1))

        return action.item()
    


    def act_top_k(self, state_np, allowed_action_ids):
        """
        Sample an action ONLY from a subset of allowed action indices.

        Args:
            state_np: np.ndarray, shape (F,) or (B,F)
            allowed_action_ids: list/tuple/1D tensor of allowed global action indices
                e.g. [160, 161, 162]  (optionally include do-nothing id)

        Returns:
            int: sampled global action id (one of allowed_action_ids)
        """
        x = torch.from_numpy(state_np).float().to(self.value_layer.weight.device)
        x = self._sanitize(x)

        if x.dim() == 1:                      # ensure [B, F]
            x = x.unsqueeze(0)

        h = self.affine(x)
        h = torch.nan_to_num(h)

        logits = self.action_layer(h)         # (B, action_dim)
        logits = torch.nan_to_num(logits)
        logits = logits - logits.max(dim=-1, keepdim=True).values

        # --------- TOP-K / MASKING ----------
        action_dim = logits.size(-1)
        device = logits.device

        if not torch.is_tensor(allowed_action_ids):
            allowed_action_ids = torch.tensor(allowed_action_ids, dtype=torch.long, device=device)
        else:
            allowed_action_ids = allowed_action_ids.to(device=device, dtype=torch.long)

        # Build mask: True for allowed actions
        allowed_mask = torch.zeros(action_dim, dtype=torch.bool, device=device)
        allowed_mask[allowed_action_ids] = True

        # Expand mask to batch
        allowed_mask_b = allowed_mask.unsqueeze(0).expand_as(logits)

        # Mask out disallowed actions
        masked_logits = logits.masked_fill(~allowed_mask_b, self.NEG_INF)

        probs = torch.softmax(masked_logits, dim=-1)

        # Safety: if something went wrong (e.g., empty allowed set), fallback to original probs
        if (not torch.isfinite(probs).all()) or torch.any(probs.sum(dim=-1) == 0):
            probs = torch.softmax(logits, dim=-1)

        dist = Categorical(probs=probs)
        action = dist.sample()  # (B,)

        self.logprobs.append(dist.log_prob(action).squeeze(-1))
        self.state_values.append(self.value_layer(h).squeeze(-1))

        return action.item()

    

    def calculateLoss(self, gamma=0.99, value_coef=0.5, entropy_coef=0.01):
        # discounted returns
        returns = []
        g = 0.0
        for r in reversed(self.rewards):
            g = r + gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
    
        # stabilize return normalization
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-8)
    
        values = torch.stack(self.state_values).to(self.device).squeeze(-1)
        logprobs = torch.stack(self.logprobs).to(self.device)
    
        advantages = returns - values.detach()
        # advantage normalization helps a LOT with small/medium LR
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    
        policy_loss = -(logprobs * advantages).mean()
        value_loss  = F.smooth_l1_loss(values, returns)
    
        # crude entropy from logprobs; better is dist.entropy() if you also stored dist params
        entropy = -(logprobs.exp() * logprobs).mean()
    
        return policy_loss + value_coef * value_loss - entropy_coef * entropy

    
    def clearMemory(self):
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]

    def save_checkpoint(self, optimizer:optim=None, filename="actor_critic_checkpoint.pth"):
        """Save model + optimizer for exact training resumption."""
        os.makedirs("models", exist_ok=True)
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        save_path = os.path.join("models", filename)
        torch.save(checkpoint, save_path)
        print(f"[SAVE] Checkpoint saved to {save_path}")


    def load_checkpoint(self, folder_name=None, filename="actor_critic_checkpoint.pth", optimizer:optim=None, load_optimizer=True):
        """Load model + optimizer state."""
        if folder_name is not None:
            file_path = os.path.join(folder_name, filename)
        else:
            file_path = os.path.join("models", filename)
        if not os.path.exists(file_path):
            print(f"[LOAD] No checkpoint found at {file_path}")
            return False

        checkpoint = torch.load(file_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"[LOAD] Checkpoint loaded from {file_path}")
        return True