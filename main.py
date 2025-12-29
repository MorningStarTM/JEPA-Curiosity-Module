import os
import grid2op
from lightsim2grid import LightSimBackend
from G_JEPA.trainer import JepaCM
from grid2op.Reward.l2RPNSandBoxScore import L2RPNSandBoxScore


env = grid2op.make("l2rpn_case14_sandbox", 
                #reward_class=L2RPNSandBoxScore,
                backend=LightSimBackend(),
                #other_rewards={"loss": LossReward, "margin": MarginReward}
                   )
# env_name = "rte_case5_example"
# env = grid2op.make(env_name, test=True, reward_class=L2RPNSandBoxScore)

jepa_config = {
    # === Encoder (state → latent) ===
    "encoder_input_dim":  493,         # dim of obs/state vector
    "encoder_hidden_dim": 512,             # you can tune this
    "encoder_output_dim": 256,              # latent dim z_t

    # === Predictor ( [z_t, one_hot(a_t)] → predict next latent ) ===
    # input = encoder_output_dim + action_dim
    "pred_input_dim":    256+178,   # must be encoder_output_dim + action_dim
    "pred_hidden_dim":   512,
    "pred_output_dim":   256,              # should match encoder_output_dim

    # === Action space ===
    "action_dim":        178,         # number of discrete actions

    # === Optimizer & EMA ===
    "learning_rate":     1e-3,
    "tau":               0.99,            # EMA factor for target encoder
    "final_jepa":"case5_jepa.pt",
    "load_model":True,
    "pred_num_blocks" :24,
    "pred_mlp_dim":512,
    "jepa_checkpoint_path":"G_JEPA/models"
}



actor_config = {
    "input_dim":493, #env.observation_space.shape.sum(),
    "action_dim":178,
    "gamma": 0.99,
    "learning_rate": 0.0003,
    "betas": (0.9, 0.999),
    "update_freq": 512,
    "save_path":"G_JEPA/models",
    'checkpoint_path':"case5_actor_critic.pt",
    'episodes': 5000,
    'num_blocks' : 12,
    'max_ep_len':8063,
    'icm_lr':1e-4,
    'beta':1e-4,
    'alpha':1e-4,
    'batch_size':256,
    'intrinsic_reward_weight':1,
    "episode_path":"G_JEPA/episode_length",
    "load_model":True,
    'actor_checkpoint_path':"models"
}


def jepa_training(start=0, end=15):
    trainer = JepaCM(env, jepa_config, actor_config)
    trainer.train(start=0, end=15)


def jepa_attack_training(start, end):
    lines = [1, 3, 2, 5, 7, 8, 11, 19, 16]
    trainer = JepaCM(env, jepa_config, actor_config)
    trainer.train_attacked_env(start=start, end=end, line2attack=lines)


if __name__ == "__main__":
    jepa_attack_training(0, 15)
    #jepa_attack_training(0, 10)