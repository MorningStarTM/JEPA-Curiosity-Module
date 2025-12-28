import os
import grid2op
from lightsim2grid import LightSimBackend
from G_JEPA.trainer import ActorCriticTrainer

env = grid2op.make("l2rpn_case14_sandbox", 
                #reward_class=L2RPNSandBoxScore,
                backend=LightSimBackend(),
                #other_rewards={"loss": LossReward, "margin": MarginReward}
                   )





actor_config = {
    "input_dim":493, #env.observation_space.shape.sum(),
    "action_dim":178,
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "betas": (0.9, 0.999),
    "update_freq": 512,
    "save_path":"G_JEPA\models",
    'checkpoint_path':"final_actor_critic.pt",
    'episodes': 5000,
    'num_blocks' : 24,
    'max_ep_len':8063,
    'icm_lr':1e-4,
    'beta':1e-4,
    'alpha':1e-4,
    'batch_size':256,
    'intrinsic_reward_weight':1,
    "episode_path":"G_JEPA\episode_length",
    "load_model":True,
    'actor_checkpoint_path':"models"
}

trainer = ActorCriticTrainer(env, actor_config)
trainer.train()