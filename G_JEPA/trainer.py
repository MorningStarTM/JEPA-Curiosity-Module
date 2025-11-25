import torch
import os
import numpy as np
from G_JEPA.agents import ActorCriticUP
from G_JEPA.jepa import JEPA
from G_JEPA.utils.logger import logger
from G_JEPA.utils.converter import ActionConverter
from G_JEPA.utils.utils import save_episode_rewards
from torch.utils.tensorboard import SummaryWriter



class JepaCM:
    def __init__(self, env, jepa_config, actor_config):
        self.jepa_config = jepa_config
        self.actor_config = actor_config
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = ActorCriticUP()
        self.jepa = JEPA(jepa_config)

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=actor_config['learning_rate'])

        log_dir = actor_config.get("log_dir", "runs/jepa")
        self.jepa_writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        self.converter = ActionConverter(self.env)

        self.episode_rewards = []
        self.episode_lenths = []
        self.episode_reasons = []
        self.episode_path = self.actor_config['episode_path']
        os.makedirs(self.episode_path, exist_ok=True)
        logger.info(f"Episode path : {self.episode_path}")


    def train(self):
        logger.info("""======================================================= \n
                                    Train function Invoke \n
                       =======================================================""")
        
        running_reward = 0
        for i_episode in range(0, self.actor_config['episodes']):
            #logger.info(f"Episode : {i_episode}")
            obs = self.env.reset()
            done = False
            episode_total_reward = 0

            for t in range(self.actor_config['max_ep_len']):
                action = self.agent(obs.to_vect())
                obs_, reward, done, _ = self.env.step(self.converter.act(action))

                intrinsic_reward = self.jepa.intrinsic_reward(obs.to_vect(), action, obs_.to_vect())
                self.jepa.memory.remember(obs.to_vect(), int(action), obs_.to_vect())

                total_reward = reward + intrinsic_reward

                self.agent.rewards.append(total_reward)
                episode_total_reward += total_reward
                obs = obs_

                if done:
                    break

                self.episode_rewards.append(episode_total_reward)  
            self.episode_lenths.append(t + 1)
            # Updating the policy :
            self.optimizer.zero_grad()
            self.jepa.optimizer.zero_grad()

            jepa_loss = self.jepa.learn()
            policy_loss = self.agent.calculateLoss(self.actor_config['gamma'])
            total_loss = policy_loss + jepa_loss

            # log JEPA loss
            self.jepa_writer.add_scalar("loss/jepa", jepa_loss.item(), self.global_step)
            self.global_step += 1


            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.jepa.encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(self.jepa.predictor.parameters(), 1.0)
            self.optimizer.step()     
            self.jepa.optimizer.step()

            self.jepa.soft_update()

            self.agent.clearMemory()
            self.jepa.memory.clear_memory()

            # saving the model if episodes > 999 OR avg reward > 200 
            if i_episode != 0 and i_episode % 1000 == 0:
                self.agent.save_checkpoint(filename="final_actor_critic.pt")    
                self.jepa.save_checkpoint(filename="final_icm.pt")
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, episode_total_reward))
                running_reward = 0

        save_episode_rewards(self.episode_rewards, save_dir="G_JEPA\\episode_reward", filename="final_actor_critic_reward.npy")
        np.save(os.path.join(self.episode_path, "final_actor_critic_lengths.npy"),
                np.array(self.episode_lenths, dtype=np.int32)) 
        logger.info(f"reward saved at G_JEPA\\episode_reward")
        self.jepa_writer.close()

        