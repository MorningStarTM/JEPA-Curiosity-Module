import torch
import os
import re
import numpy as np
import random
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
        self.danger = 0.9
        self.thermal_limit = self.env._thermal_limit_a
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = ActorCriticUP(actor_config)
        
        self.agent.param_counts()

        self.jepa = JEPA(jepa_config)
        self.jepa.print_jepa_params()

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=actor_config['learning_rate'])

        if actor_config['load_model']:
            self.agent.load_checkpoint(optimizer=self.optimizer, folder_name=actor_config['actor_checkpoint_path'], filename=actor_config['checkpoint_path'])
            logger.info(f"ActorCritic Model Loaded from {actor_config['actor_checkpoint_path']}/{actor_config['checkpoint_path']}")
        
        if jepa_config['load_model']:
            self.jepa.load_checkpoint(folder_name=jepa_config['jepa_checkpoint_path'], filename=jepa_config['final_jepa'])
            logger.info(f"JEPA Model Loaded from {jepa_config['jepa_checkpoint_path']}/{jepa_config['final_jepa']}")

         # Tensorboard Writer

        log_dir = actor_config.get("log_dir", "runs/jepa")
        self.jepa_writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

        self.converter = ActionConverter(self.env)

        self.episode_rewards = []
        self.episode_lenths = []
        self.episode_reasons = []
        self.agent_actions = []
        self.actor_loss = []
        self.jepa_loss = []
        self.episode_path = self.actor_config['episode_path']
        os.makedirs(self.episode_path, exist_ok=True)
        logger.info(f"Episode path : {self.episode_path}")

    
    def is_safe(self, obs):
        
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False
        return True
    
    def make_chronic_regex(self, start, end):
        ids = "|".join(f"{i:04d}" for i in range(start, end+1))
        return rf".*({ids}).*"
    

    def train(self, start, end):
        logger.info("""======================================================= \n
                                    Train function Invoke \n
                       =======================================================""")
        
        regex = self.make_chronic_regex(start=start, end=end)
        self.env.chronics_handler.set_filter(lambda p, regex=regex: re.match(regex, p) is not None)
        self.env.chronics_handler.reset()

        running_reward = 0
        for i_episode in range(0, self.actor_config['episodes']):
            #logger.info(f"Episode : {i_episode}")
            obs = self.env.reset()
            done = False
            episode_total_reward = 0

            for t in range(self.actor_config['max_ep_len']):
                is_safe = self.is_safe(obs)

                if not is_safe:
                    action = self.agent(obs.to_vect())
                    self.agent_actions.append(int(action))
                    grid_action = self.converter.act(action)
                else:
                    grid_action = self.env.action_space({})

                obs_, reward, done, _ = self.env.step(grid_action)

                

                if not is_safe:
                    intrinsic_reward = self.jepa.intrinsic_reward(obs.to_vect(), action, obs_.to_vect())
                    self.jepa.memory.remember(obs.to_vect(), int(action), obs_.to_vect())

                    total_reward = reward + intrinsic_reward
                    self.agent.rewards.append(total_reward)
                
                else:
                    total_reward = reward

                
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
            self.actor_loss.append(policy_loss.item())
            self.jepa_loss.append(jepa_loss.item())
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
                self.agent.save_checkpoint(optimizer=self.optimizer, filename="final_actor_critic.pt")    
                self.jepa.save_checkpoint(filename="final_jepa.pt")
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, episode_total_reward))
                running_reward = 0

        save_episode_rewards(self.episode_rewards, save_dir="G_JEPA\\episode_reward", filename="final_actor_critic_reward.npy")
        np.save(os.path.join(self.episode_path, "final_actor_critic_lengths.npy"),
                np.array(self.episode_lenths, dtype=np.int32)) 
        np.save(os.path.join(self.episode_path, "actor_critic_actions.npy"), np.array(self.agent_actions, dtype=np.int32))
        np.save(os.path.join(self.episode_path, "actor_critic_loss.npy"), np.array(self.actor_loss, dtype=np.float32))
        np.save(os.path.join(self.episode_path, "jepa_loss.npy"), np.array(self.jepa_loss, dtype=np.float32))
        logger.info(f"reward saved at G_JEPA\\episode_reward")
        self.jepa_writer.close()

        

    def train_attacked_env(self, start, end, line2attack, max_attack_step=72):
        """
        Train the JEPA + ActorCritic agent on an *attacked* environment.

        - First: choose a line from `line2attack` and a random attack time step.
        - Fast-forward chronics to that step and disconnect the chosen line.
        - Then: run the usual JEPA + RL training loop from that attacked state.
        """

        logger.info("""======================================================= \n
                                Train_Attacked_Env function Invoke \n
                    =======================================================""")

        # keep same chronic filtering logic as in `train`
        regex = self.make_chronic_regex(start=start, end=end)
        self.env.chronics_handler.set_filter(lambda p, regex=regex: re.match(regex, p) is not None)
        self.env.chronics_handler.reset()

        running_reward = 0.0

        for i_episode in range(self.actor_config['episodes']):
            # --- 1) Pick attack (line + time) ----------------------------------
            line_to_disconnect = int(random.choice(line2attack))
            attack_step = random.randint(0, max_attack_step)

            obs = self.env.reset()
            done = False
            episode_total_reward = 0.0

            # --- 2) Fast-forward to attack_step --------------------------------
            if attack_step > 0:
                # go to attack_step-1, then do-nothing step to reach attack_step
                self.env.fast_forward_chronics(attack_step - 1)
                obs, reward, done, _ = self.env.step(self.env.action_space({}))
                if done:
                    # scenario already dead before attack, skip episode
                    continue

            # --- 3) Apply the line attack --------------------------------------
            # disconnect only the chosen line, keep others unchanged
            new_line_status_array = np.zeros_like(obs.rho, dtype=np.int32)
            new_line_status_array[line_to_disconnect] = -1
            attack_action = self.env.action_space({"set_line_status": new_line_status_array})
            obs, reward, done, _ = self.env.step(attack_action)

            if done:
                # attack itself caused blackout: nothing useful to learn here
                continue

            # --- 4) RL control after attack (same logic as `train`) ------------
            # t starts from attack_step to keep meaning of 'time' consistent
            for t in range(attack_step, self.actor_config['max_ep_len']):
                is_safe = self.is_safe(obs)

                # Agent acts only when grid is not safe
                if not is_safe:
                    action = self.agent(obs.to_vect())
                    self.agent_actions.append(int(action))
                    grid_action = self.converter.act(action)
                else:
                    grid_action = self.env.action_space({})

                obs_, reward, done, _ = self.env.step(grid_action)

                if not is_safe:
                    intrinsic_reward = self.jepa.intrinsic_reward(
                        obs.to_vect(), action, obs_.to_vect()
                    )
                    self.jepa.memory.remember(
                        obs.to_vect(), int(action), obs_.to_vect()
                    )
                    total_reward = reward + intrinsic_reward
                    self.agent.rewards.append(total_reward)
                else:
                    total_reward = reward

                episode_total_reward += total_reward
                obs = obs_

                if done:
                    break

            # --- 5) Update JEPA + policy (same as `train`) ----------------------
            self.episode_rewards.append(episode_total_reward)
            self.episode_lenths.append(t + 1)

            self.optimizer.zero_grad()
            self.jepa.optimizer.zero_grad()

            jepa_loss = self.jepa.learn()
            policy_loss = self.agent.calculateLoss(self.actor_config['gamma'])
            self.actor_loss.append(policy_loss)
            self.jepa_loss.append(jepa_loss)

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

            # checkpointing
            if i_episode != 0 and i_episode % 1000 == 0:
                self.agent.save_checkpoint(optimizer=self.optimizer, filename="final_actor_critic.pt")
                self.jepa.save_checkpoint(filename="final_jepa.pt")

            if i_episode % 20 == 0:
                running_reward = running_reward / 20 if i_episode != 0 else episode_total_reward
                logger.info(
                    'Attacked Episode {}\tattack_line: {}\tattack_step: {}\tlength: {}\treward: {}'.format(
                        i_episode, line_to_disconnect, attack_step, t, episode_total_reward
                    )
                )
                running_reward = 0.0

        # --- 6) Save stats (same as `train`) -----------------------------------
        save_episode_rewards(self.episode_rewards, save_dir="G_JEPA\\episode_reward",
                            filename="final_actor_critic_reward_attacked.npy")
        np.save(os.path.join(self.episode_path, "final_actor_critic_lengths_attacked.npy"),
                np.array(self.episode_lenths, dtype=np.int32))
        np.save(os.path.join(self.episode_path, "actor_critic_actions_attacked.npy"),
                np.array(self.agent_actions, dtype=np.int32))
        np.save(os.path.join(self.episode_path, "actor_critic_loss_attacked.npy"),
                np.array(self.actor_loss, dtype=np.float32))
        np.save(os.path.join(self.episode_path, "jepa_loss_attacked.npy"),
                np.array(self.jepa_loss, dtype=np.float32))
        logger.info("Attacked-env rewards saved at G_JEPA\\episode_reward")
        self.jepa_writer.close()







class ActorCriticTrainer:
    def __init__(self, env, actor_config):
        self.actor_config = actor_config
        self.env = env
        self.danger = 0.9
        self.thermal_limit = self.env._thermal_limit_a
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent = ActorCriticUP(actor_config)
        
        self.agent.param_counts()

        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=actor_config['learning_rate'])

        if actor_config['load_model']:
            self.agent.load_checkpoint(optimizer=self.optimizer, folder_name=actor_config['actor_checkpoint_path'], filename=actor_config['checkpoint_path'])
            logger.info(f"ActorCritic Model Loaded from {actor_config['actor_checkpoint_path']}/{actor_config['checkpoint_path']}")

        self.converter = ActionConverter(self.env)

        self.episode_rewards = []
        self.episode_lenths = []
        self.episode_reasons = []
        self.agent_actions = []
        self.actor_loss = []
        self.episode_path = self.actor_config['episode_path']
        os.makedirs(self.episode_path, exist_ok=True)
        logger.info(f"Episode path : {self.episode_path}")



    def is_safe(self, obs):
        
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger - 0.05) or ratio >= self.danger:
                return False
        return True
    
    
    def train(self):
        running_reward = 0
        actions = []
        for i_episode in range(0, self.actor_config['episodes']):
            logger.info(f"Episode : {i_episode}")
            obs = self.env.reset()
            done = False
            episode_total_reward = 0

            for t in range(self.actor_config['max_ep_len']):
                is_safe = self.is_safe(obs)

                if not is_safe:
                    action = self.agent(obs.to_vect())
                    actions.append(action)
                    grid_action = self.converter.act(action)
                else:
                    grid_action = self.env.action_space({})
                obs_, reward, done, _ = self.env.step(grid_action)

                if not is_safe:
                    self.agent.rewards.append(reward)

                episode_total_reward += reward
                obs = obs_

                if done:
                    break

            logger.info(f"Episode {i_episode} reward: {episode_total_reward}")  
            self.episode_rewards.append(episode_total_reward)  
            self.episode_lenths.append(t + 1)
            # Updating the policy :
            self.optimizer.zero_grad()
            loss = self.agent.calculateLoss(self.actor_config['gamma'])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), max_norm=0.5)
            self.optimizer.step()        
            self.agent.clearMemory()

            # saving the model if episodes > 999 OR avg reward > 200 
            if i_episode != 0 and i_episode % 1000 == 0:
                self.agent.save_checkpoint(optimizer=self.optimizer, filename="fine_tuned_actor_critic.pt")    
           
            
            if i_episode % 20 == 0:
                running_reward = running_reward/20
                logger.info('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, episode_total_reward))
                running_reward = 0
            
            survival_steps = t + 1          # because t is 0-indexed
            self.episode_lenths.append(survival_steps)

        save_episode_rewards(self.episode_rewards, save_dir="ICM\\episode_reward", filename="fine_tuned_actor_critic_reward.npy")
        np.save(os.path.join(self.episode_path, "fine_tuned_actor_critic_lengths.npy"),
                np.array(self.episode_lenths, dtype=np.int32)) 
        np.save(os.path.join(self.episode_path, "fine_tuned_actor_critic_actions.npy"), np.array(actions, dtype=np.int32))
        np.save(os.path.join(self.episode_path, "fine_tuned_actor_critic_loss.npy"), np.array(self.actor_loss, dtype=np.float32))
        logger.info(f"reward saved at ICM\\episode_reward")
        os.makedirs("ICM\\episode_reward", exist_ok=True)
        np.save("ICM\\episode_reward\\fine_tuned_actor_critic_steps.npy", np.array(self.episode_lenths, dtype=int))
