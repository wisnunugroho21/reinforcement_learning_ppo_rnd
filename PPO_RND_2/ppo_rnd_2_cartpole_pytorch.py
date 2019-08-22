import gym
from gym.envs.registration import register
    
import torch
import torch.nn as nn
from torch.distributions import Categorical
import torchvision

import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
      
class PPO_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO_Model, self).__init__()   

        # Actor
        self.actor_layer = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ELU(),
                nn.Linear(64, 64),
                nn.ELU(),
                nn.Linear(64, action_dim),
                nn.Softmax(-1)
              ).float().to(device)
        
        # Intrinsic Critic
        self.value_in_layer = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ELU(),
                nn.Linear(64, 64),
                nn.ELU(),
                nn.Linear(64, 1)
              ).float().to(device)
        
        # External Critic
        self.value_ex_layer = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ELU(),
                nn.Linear(64, 64),
                nn.ELU(),
                nn.Linear(64, 1)
              ).float().to(device)
        
    # Init wieghts to make training faster
    # But don't init weight if you load weight from file
    def lets_init_weights(self):      
        self.actor_layer.apply(self.init_weights)
        self.value_in_layer.apply(self.init_weights)
        self.value_ex_layer.apply(self.init_weights)
        
    def init_weights(self, m):
        for name, param in m.named_parameters():
            if 'bias' in name:
               nn.init.constant_(param, 0.01)
            elif 'weight' in name:
                nn.init.kaiming_uniform_(param, mode = 'fan_in', nonlinearity = 'leaky_relu')

    def init_memory_weights(self, m):
        for name, param in m.named_parameters():
            if 'bias' in name:
               nn.init.constant_(param, 0.01)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('tanh'))
        
    def forward(self, state, is_act = False):
        if is_act:         
            return self.actor_layer(state)
        else:
            return self.actor_layer(state), self.value_in_layer(state), self.value_ex_layer(state)
      
class RND_predictor_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RND_predictor_Model, self).__init__()

        # State Predictor
        self.state_predict_layer = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ELU(),
                nn.Linear(64, 64),
                nn.ELU(),
                nn.Linear(64, 4)
              ).float().to(device)

    def init_state_predict_weights(self, m):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.01)
            elif 'weight' in name:
                nn.init.constant_(param, 1)

    def lets_init_weights(self):      
        self.state_predict_layer.apply(self.init_state_predict_weights)

    def forward(self, state):
        return self.state_predict_layer(state)

class RND_target_Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(RND_target_Model, self).__init__()

        # State Target
        self.state_target_layer = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ELU(),
                nn.Linear(64, 64),
                nn.ELU(),
                nn.Linear(64, 4)
              ).float().to(device)
              
    def init_state_target_weights(self, m):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1)
            elif 'weight' in name:
                nn.init.constant_(param, 0.01)

    def lets_init_weights(self):      
        self.state_target_layer.apply(self.init_state_target_weights)

    def forward(self, state):
        return self.state_target_layer(state)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []     
        self.next_states = []
        self.observation = []
        self.mean_obs = 0
        self.std_obs = 0
        
    def save_eps(self, state, reward, next_states, done):
        self.rewards.append(reward)
        self.states.append(state)
        self.dones.append(done)
        self.next_states.append(next_states)
        
    def save_actions(self, action):
        self.actions.append(action)
        
    def save_logprobs(self, logprob):
        self.logprobs.append(logprob)
        
    def save_observation(self, obs):
        self.observation.append(obs)
        
    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        
    def clearObs(self):
        del self.observation[:]
        
class Utils:
    def __init__(self):
        self.gamma = 0.95
        self.lam = 0.99

    # Categorical Distribution is used for Discrete Action Environment
    # The neural network output the probability of actions (Stochastic policy), then pass it to Categorical Distribution
    
    def sample(self, datas):
        distribution = Categorical(datas)      
        return distribution.sample().float().to(device)
        
    def entropy(self, datas):
        distribution = Categorical(datas)            
        return distribution.entropy().float().to(device)
      
    def logprob(self, datas, value_data):
        distribution = Categorical(datas)
        return distribution.log_prob(value_data).float().to(device)      
      
    def normalize(self, data, mean = None, std = None):
        if isinstance(mean, torch.Tensor) and isinstance(std, torch.Tensor):
            data_normalized = (data - mean) / std            
        else:
            data_normalized = (data - torch.mean(data)) / torch.std(data)

        return data_normalized
      
    def to_numpy(self, datas):
        if torch.cuda.is_available():
            datas = datas.cpu().detach().numpy()
        else:
            datas = datas.detach().numpy()            
        return datas        
      
    def discounted(self, datas):
        # Discounting future reward        
        discounted_datas = torch.zeros_like(datas)
        running_add = 0
        
        for i in reversed(range(len(datas))):
            running_add = running_add * self.gamma + datas[i]
            discounted_datas[i] = running_add
            
        return discounted_datas
      
    def q_values(self, reward, next_value, done, value_function):
        # Finding Q Values
        # Q = R + V(St+1)
        q_values = reward + (1 - done) * self.gamma * next_value           
        return q_values
      
    def compute_GAE(self, values, rewards, next_value, done):
        # Computing general advantages estimator
        gae = 0
        returns = []
        
        for step in reversed(range(len(rewards))):   
            delta = rewards[step] + self.gamma * next_value[step] * (1 - done[step]) - values[step]
            gae = delta + self.lam * gae
            returns.append(gae.detach())
            
        return torch.stack(returns)

    def prepro(self, I):
        # Crop the image and convert it to Grayscale
        # For more information : https://medium.com/@dhruvp/how-to-write-a-neural-network-to-play-pong-from-scratch-956b57d4f6e0

        """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
        I = I[35:195] # crop
        #I = np.dot(I[...,:3], [0.2989, 0.5870, 0.1140])
        I = I[::2,::2, 0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        
        X = I.astype(np.float32).ravel() # Combine items in 1 array 
        return X
        
class Agent:  
    def __init__(self, state_dim, action_dim):        
        self.policy_clip = 0.1 
        self.value_clip = 1      
        self.entropy_coef = 0.01
        self.vf_loss_coef = 1

        self.PPO_epochs = 3
        self.RND_epochs = 3
        
        self.ex_advantages_coef = 2
        self.in_advantages_coef = 1
        
        self.policy = PPO_Model(state_dim, action_dim)
        self.policy_old = PPO_Model(state_dim, action_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = 0.0001) 

        self.rnd_predict = RND_predictor_Model(state_dim, action_dim)
        self.rnd_predict_optimizer = torch.optim.Adam(self.rnd_predict.parameters(), lr = 0.0001)
        self.rnd_target = RND_target_Model(state_dim, action_dim)

        self.memory = Memory()
        self.utils = Utils()        
        
    def save_eps(self, state, reward, next_states, done):
        self.memory.save_eps(state, reward, next_states, done)
        
    def save_observation(self, obs):
        self.memory.save_observation(obs)

    # Loss for RND 
    def get_rnd_loss(self, obs, mean_obs, std_obs):
        obs = self.utils.normalize(obs, mean_obs, std_obs)

        state_pred = self.rnd_predict(obs)
        state_target = self.rnd_target(obs)
        
        # Don't update target state value
        state_target = state_target.detach()
        
        # Mean Squared Error Calculation between state and predict
        forward_loss = (state_target - state_pred).pow(2).mean()
        return forward_loss

    # Loss for PPO
    def get_loss(self, mean_obs, std_obs, old_states, old_actions, rewards, old_next_states, dones):      
        action_probs, in_value, ex_value  = self.policy(old_states)  
        old_action_probs, in_old_value, ex_old_value = self.policy_old(old_states)
        _, next_in_value, next_ex_value = self.policy(old_next_states)
        
        obs = self.utils.normalize(old_next_states, mean_obs, std_obs)
        state_pred = self.rnd_predict(obs)
        state_target = self.rnd_target(obs)
        
        # Don't update old value
        old_action_probs = old_action_probs.detach()
        in_old_value = in_old_value.detach()
        ex_old_value = ex_old_value.detach()

        # Don't update rnd value
        state_target = state_target.detach()
        state_pred = state_pred.detach()        

        # Don't update next value
        next_in_value = next_in_value.detach()
        next_ex_value = next_ex_value.detach()
                
        # Getting entropy from the action probability
        dist_entropy = self.utils.entropy(action_probs).mean()

        # Getting external general advantages estimator
        external_rewards = rewards.detach()
        external_advantage = self.utils.compute_GAE(ex_value, rewards, next_ex_value, dones).detach()
                    
        # Computing internal reward, then getting internal general advantages estimator
        intrinsic_rewards = (state_target - state_pred).pow(2).mean(-1).detach()
        intrinsic_advantage = self.utils.compute_GAE(in_value, intrinsic_rewards, next_in_value, dones)
        intrinsic_advantage = self.utils.normalize(intrinsic_advantage).detach()
        
        # Getting overall advantages
        advantages = (self.ex_advantages_coef * external_advantage + self.in_advantages_coef * intrinsic_advantage).detach()
        
        # Getting External critic loss by using Clipped critic value
        ex_vpredclipped = ex_old_value + torch.clamp(ex_value - ex_old_value, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        ex_vf_losses1 = (external_rewards - ex_value).pow(2) # Mean Squared Error
        ex_vf_losses2 = (external_rewards - ex_vpredclipped).pow(2) # Mean Squared Error
        critic_ext_loss = torch.min(ex_vf_losses1, ex_vf_losses2).mean()

        # Getting Intrinsic critic loss
        critic_int_loss = (intrinsic_rewards - in_value).pow(2).mean()
        
        # Getting overall critic loss
        critic_loss = critic_ext_loss + critic_int_loss

        # Finding the ratio (pi_theta / pi_theta__old):  
        logprobs = self.utils.logprob(action_probs, old_actions) 
        old_logprobs = self.utils.logprob(old_action_probs, old_actions).detach()
        
        # Finding Surrogate Loss:
        ratios = torch.exp(logprobs - old_logprobs) # ratios = old_logprobs / logprobs
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
        pg_loss = torch.min(surr1, surr2).mean()  
        
        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss and 
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss 
                
        return loss       
      
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action_probs = self.policy_old(state, is_act = True)

        # Sample the action
        action = self.utils.sample(action_probs)       
        
        self.memory.save_actions(action)         
        return action.item() 

    # Update the RND part (the state and predictor)
    def update_rnd(self):
        # Convert list in tensor
        obs = torch.FloatTensor(self.memory.observation).to(device).detach()
        mean_obs = self.memory.mean_obs.detach()
        std_obs = self.memory.std_obs.detach()
        
        # Optimize predictor for K epochs:
        for epoch in range(self.RND_epochs):                        
            loss = self.get_rnd_loss(obs, mean_obs, std_obs)  
            
            self.rnd_predict_optimizer.zero_grad()
            loss.backward()                    
            self.rnd_predict_optimizer.step() 

        # Clear the observation
        self.memory.clearObs()        
        
    # Update the PPO part (the actor and value)
    def update_ppo(self):        
        length = len(self.memory.states)

        # Convert list in tensor
        old_states = torch.FloatTensor(self.memory.states).to(device).detach()
        old_actions = torch.FloatTensor(self.memory.actions).to(device).detach()
        old_next_states = torch.FloatTensor(self.memory.next_states).to(device).detach()

        # Convert list to tensor and change the dimensions
        dones = torch.FloatTensor(self.memory.dones).view(length, 1).view(length, 1).to(device).detach() 
        rewards = torch.FloatTensor(self.memory.rewards).view(length, 1).view(length, 1).to(device).detach()

        # Getting value
        mean_obs = self.memory.mean_obs.detach()
        std_obs = self.memory.std_obs.detach()
                
        # Optimize policy for K epochs:
        for epoch in range(self.PPO_epochs):
            loss = self.get_loss(mean_obs, std_obs, old_states, old_actions, rewards, old_next_states, dones)          
                        
            self.policy_optimizer.zero_grad()
            loss.backward()                    
            self.policy_optimizer.step() 

        # Clear state, action, reward in memory    
        self.memory.clearMemory()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def save_weights(self):
        torch.save(self.policy.state_dict(), '/test/My Drive/RL_Pong_PPO_RND/actor_pong_ppo_rnd.pth')
        torch.save(self.policy_old.state_dict(), '/test/My Drive/RL_Pong_PPO_RND/old_actor_pong_ppo_rnd.pth')
        torch.save(self.rnd_predict.state_dict(), '/test/My Drive/RL_Pong_PPO_RND/rnd_predict_pong_ppo_rnd.pth')
        torch.save(self.rnd_target.state_dict(), '/test/My Drive/RL_Pong_PPO_RND/rnd_target_pong_ppo_rnd.pth')
        
    def load_weights(self):
        self.policy.load_state_dict(torch.load('/test/My Drive/RL_Pong_PPO_RND/actor_pong_ppo_rnd.pth'))        
        self.policy_old.load_state_dict(torch.load('/test/My Drive/RL_Pong_PPO_RND/old_actor_pong_ppo_rnd.pth'))
        self.rnd_predict.load_state_dict(torch.load('/test/My Drive/RL_Pong_PPO_RND/rnd_predict_pong_ppo_rnd.pth'))        
        self.rnd_target.load_state_dict(torch.load('/test/My Drive/RL_Pong_PPO_RND/rnd_target_pong_ppo_rnd.pth'))    
        
    def lets_init_weights(self):
        self.policy.lets_init_weights()
        self.policy_old.lets_init_weights()
        
def plot(datas):
    print('----------')
    
    plt.plot(datas)
    plt.plot()
    plt.xlabel('Episode')
    plt.ylabel('Datas')
    plt.show()
    
    print('Max :', np.max(datas))
    print('Min :', np.min(datas))
    print('Avg :', np.mean(datas))

def run_episode(env, agent, state_dim, render, t_rnd, training_mode, n_rnd_update):
    utils = Utils()
    ############################################
    state = env.reset() 
    done = False
    total_reward = 0
    t = 0
    ############################################
    
    while not done:
        # Running policy_old:
        action = int(agent.act(state))
        state_n, reward, done, info = env.step(action)
        
        t += 1
        t_rnd += 1                           
        total_reward += reward  

        # There is a bugs in cartpole env which always giving same rewards, even if the state will make env terminate (done = true)
        # So, i tweak the reward system
        # reward will negative if the state will make env terminate (done = true)
        reward = -100 if done and t < 200 else 1
          
        if training_mode:
            agent.save_eps(state, reward, state_n, done) 
            agent.save_observation(state_n)
            
        state = state_n     
        
        if training_mode:
            if t_rnd == n_rnd_update:
                agent.update_rnd()
                print('RND has been updated')
                t_rnd = 0
        
        if render:
            env.render()
        if done:
            return total_reward, t, t_rnd

def run_inits_episode(env, agent, render, n_init_episode):
    utils = Utils()
    obs = []
    ############################################
    state_val = env.reset()

    for i in range(n_init_episode):
        action = env.action_space.sample()
        state_n, _, done, _ = env.step(action)
        obs.append(state_n)

        if render:
            env.render()

        if done:
            env.reset()

    obs = torch.FloatTensor(obs).to(device).detach()
    agent.memory.mean_obs = torch.mean(obs, 0, True)
    agent.memory.std_obs = torch.std(obs, 0, True)

    print('mean_obs : ', agent.memory.mean_obs)
    print('std_obs : ', agent.memory.std_obs)

    return agent
    
def main():
    ############## Hyperparameters ##############
    using_google_drive = False # If you using Google Colab and want to save the agent to your GDrive, set this to True
    load_weights = False # If you want to load the agent, set this to True
    save_weights = False # If you want to save the agent, set this to True
    training_mode = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold = 195 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    
    render = False # If you want to display the image. Turn this off if you run this in Google Collab
    n_update = 1 # How many episode before you update the Policy
    n_plot_batch = 100 # How many episode you want to plot the result
    n_rnd_update = 128 # How many episode before you update the RND
    n_episode = 1000 # How many episode you want to run
    n_init_episode = 2048

    mean_obs = None
    std_obs = None
    #############################################         
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
        
    utils = Utils()     
    agent = Agent(state_dim, action_dim)  
    ############################################# 
    
    if using_google_drive:
        from google.colab import drive
        drive.mount('/test')
    
    if load_weights:
        agent.load_weights()
        print('Weight Loaded')
    else :
        agent.lets_init_weights()
        print('Init Weight')
    
    if torch.cuda.is_available() :
        print('Using GPU')

    #############################################

    if training_mode:
        if mean_obs and std_obs:
            agent.memory.mean_obs = torch.FloatTensor([mean_obs]).to(device).detach()
            agent.memory.std_obs = torch.FloatTensor([std_obs]).to(device).detach()
            print('Init obs normalization parameter manually')
        else:
            print('Start running initial episode')
            agent = run_inits_episode(env, agent, render, n_init_episode)
            print('Finish running initial episode')

    #############################################
    
    rewards = []   
    batch_rewards = []
    batch_solved_reward = []
    
    times = []
    batch_times = []
    
    t_rnd = 0

    #############################################
    
    for i_episode in range(1, n_episode):
        total_reward, time, t_rnd = run_episode(env, agent, state_dim, render, t_rnd, training_mode, n_rnd_update)
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, total_reward, time))
        batch_rewards.append(total_reward)
        batch_times.append(time)       
        
        if training_mode:
            # update after n episodes
            if i_episode % n_update == 0 and i_episode != 0:
                agent.update_ppo()
                #print('Agent has been updated')

                if save_weights:
                    agent.save_weights()
                    #print('Weights saved')
                    
        if reward_threshold:
            if len(batch_solved_reward) == 100:            
                if np.mean(batch_solved_reward) >= reward_threshold :              
                    for reward in batch_times:
                        rewards.append(reward)

                    for time in batch_rewards:
                        times.append(time)                    

                    print('You solved task after {} episode'.format(len(rewards)))
                    break

                else:
                    del batch_solved_reward[0]
                    batch_solved_reward.append(total_reward)

            else:
                batch_solved_reward.append(total_reward)
            
        if i_episode % n_plot_batch == 0 and i_episode != 0:
            # Plot the reward, times for every n_plot_batch
            plot(batch_rewards)
            plot(batch_times)
            
            for reward in batch_times:
                rewards.append(reward)
                
            for time in batch_rewards:
                times.append(time)
                
            batch_rewards = []
            batch_times = []

            print('========== Cummulative ==========')
            # Plot the reward, times for every episode
            plot(rewards)
            plot(times)
            
    print('========== Final ==========')
     # Plot the reward, times for every episode
    plot(rewards)
    plot(times) 

    #############################################

if __name__ == '__main__':
    main()