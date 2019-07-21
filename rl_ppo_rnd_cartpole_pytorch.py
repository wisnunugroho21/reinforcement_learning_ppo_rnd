from google.colab import drive

import gym
from gym.envs.registration import register
    
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
      
class Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Model, self).__init__()
        
        # Actor
        self.actor_layer = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Softmax(-1)
              ).float().to(device)
        
        # Intrinsic Critic
        self.value_in_layer = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
              ).float().to(device)
        
        # External Critic
        self.value_ex_layer = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
              ).float().to(device)
        
        # State Predictor
        self.state_predict_layer = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32,1)
              ).float().to(device)
        
        # State Target
        self.state_target_layer = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
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
                nn.init.kaiming_normal_(param, mode = 'fan_in', nonlinearity = 'relu')
        
    def forward(self, state):
        return self.actor_layer(state), self.value_in_layer(state), self.value_ex_layer(state), self.state_predict_layer(state), self.state_target_layer(state)
      
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []     
        self.next_states = []
        self.observation = []
        
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
        self.lam = 0.95

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
      
    def normalize(self, data):
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
        
class Agent:  
    def __init__(self, state_dim, action_dim):        
        self.eps_clip = 0.2
        self.K_epochs = 5
        self.entropy_coef = 0.001
        self.vf_loss_coef = 1
        self.target_kl = 0.01
        
        self.ex_advantages_coef = 2
        self.in_advantages_coef = 1
        self.ex_critic_coef = 1
        self.in_critic_coef = 1
        
        self.policy = Model(state_dim, action_dim)
        self.policy_old = Model(state_dim, action_dim) 
        
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = 0.001) 
        self.memory = Memory()
        self.utils = Utils()        
        
    def save_eps(self, state, reward, next_states, done):
        self.memory.save_eps(state, reward, next_states, done)
        
    def save_observation(self, obs):
        self.memory.save_observation(obs)
        
    # Loss for RND 
    def get_rnd_loss(self, old_states):
        _, _, _, state_pred, state_target = self.policy(old_states)
        
        # Don't update target state value
        state_target = state_target.detach()
        
        # Mean Squared Error Calculation between state and predict
        forward_loss = (state_target - state_pred).pow(2).mean()
        return forward_loss

    # Loss for PPO
    def get_loss(self, old_states, old_actions, rewards, old_next_states, dones):      
        action_probs, in_value, ex_value, state_pred, state_target = self.policy(old_states)  
        old_action_probs, in_old_value, ex_old_value, _, _ = self.policy_old(old_states)
        _, next_in_value, next_ex_value, _, _ = self.policy_old(old_next_states)
        
        # Don't update old value
        old_action_probs = old_action_probs.detach()
        in_old_value = in_old_value.detach()
        ex_old_value = ex_old_value.detach()
                
        # Getting entropy from the action probability
        dist_entropy = self.utils.entropy(action_probs).mean()
        
        # Getting external general advantages estimator
        external_rewards = rewards.detach()
        external_advantage = self.utils.compute_GAE(ex_value, rewards, next_ex_value, dones).detach()
                    
        # Computing internal reward, then getting internal general advantages estimator
        intrinsic_rewards = (state_target - state_pred).pow(2)
        intrinsic_advantage = self.utils.compute_GAE(in_value, intrinsic_rewards, next_in_value, dones).detach()
        
        # Getting overall advantages
        advantages = (self.ex_advantages_coef * external_advantage + self.in_advantages_coef * intrinsic_advantage).detach()
        
        # Finding Intrinsic Value Function Loss by using Clipped Rewards Value
        in_vpredclipped = in_old_value + torch.clamp(in_value - in_old_value, -self.eps_clip, self.eps_clip) # Minimize the difference between old value and new value
        in_vf_losses1 = (intrinsic_rewards - in_value).pow(2) # Mean Squared Error
        in_vf_losses2 = (intrinsic_rewards - in_vpredclipped).pow(2) # Mean Squared Error
        critic_int_loss = torch.min(in_vf_losses1, in_vf_losses2).mean()
        
        # Finding External Value Function Loss by using Clipped Rewards Value
        ex_vpredclipped = ex_old_value + torch.clamp(ex_value - ex_old_value, -self.eps_clip, self.eps_clip) # Minimize the difference between old value and new value
        ex_vf_losses1 = (external_rewards - ex_value).pow(2) # Mean Squared Error
        ex_vf_losses2 = (external_rewards - ex_vpredclipped).pow(2) # Mean Squared Error
        critic_ext_loss = torch.min(ex_vf_losses1, ex_vf_losses2).mean()
        
        # Getting overall critic loss
        critic_loss = critic_ext_loss + critic_int_loss

        # Finding the ratio (pi_theta / pi_theta__old):  
        logprobs = self.utils.logprob(action_probs, old_actions) 
        old_logprobs = self.utils.logprob(old_action_probs, old_actions).detach()
        
        # Finding Surrogate Loss:
        ratios = torch.exp(logprobs - old_logprobs) # ratios = old_logprobs / logprobs
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        pg_loss = torch.min(surr1, surr2).mean()       
        
        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss and 
        loss = pg_loss - (critic_loss * self.vf_loss_coef) + (dist_entropy * self.entropy_coef) 
        loss = loss * -1        
        
        # Approx KL to choose whether we must continue the gradient descent
        approx_kl = 0.5 * (logprobs - old_logprobs).pow(2).mean()
        
        return loss, approx_kl       
      
    def act(self, state):
        state = torch.FloatTensor(state).to(device)      
        action_probs, _, _, _, _ = self.policy_old(state)
        
        # Sample the action
        action = self.utils.sample(action_probs)
        self.memory.save_actions(action)   
        
        return action.item() 

    # Update the RND part (the state and predictor)
    def update_rnd(self):
        # Convert list in tensor
        old_states = torch.FloatTensor(self.memory.observation).to(device).detach()
        
        # Optimize predictor for K epochs:
        for epoch in range(self.K_epochs):            
            loss = self.get_rnd_loss(old_states)  
            
            self.policy_optimizer.zero_grad()
            loss.backward()                    
            self.policy_optimizer.step() 

        # Clear the observation
        self.memory.clearObs()
        
    # Update the PPO part (the actor and value)
    def update(self):        
        # Convert list in tensor
        old_states = torch.FloatTensor(self.memory.states).to(device).detach()
        old_actions = torch.FloatTensor(self.memory.actions).to(device).detach()
        old_next_states = torch.FloatTensor(self.memory.next_states).to(device).detach()
        dones = torch.FloatTensor(self.memory.dones).to(device).detach() 
        rewards = torch.FloatTensor(self.memory.rewards).to(device).detach()
                
        # Optimize policy for K epochs:
        for epoch in range(self.K_epochs):            
            loss, approx_kl = self.get_loss(old_states, old_actions, rewards, old_next_states, dones)          
            
            # If KL is bigger than threshold, stop update and continue to next episode
            if approx_kl > (1.5 * self.target_kl):
                print('KL greater than target. Stop update at epoch : ', epoch)
                break
            
            self.policy_optimizer.zero_grad()
            loss.backward()                    
            self.policy_optimizer.step() 

        # Clear state, action, reward in memory    
        self.memory.clearMemory()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def save_weights(self):
        torch.save(self.policy.state_dict(), '/wisnunugroho21/My Drive/cartpole/actor_pong_ppo_rnd.pth')
        torch.save(self.policy_old.state_dict(), '/wisnunugroho21/My Drive/cartpole/old_actor_pong_ppo_rnd.pth')
        
    def load_weights(self):
        self.policy.load_state_dict(torch.load('/wisnunugroho21/My Drive/cartpole/actor_pong_ppo_rnd.pth'))        
        self.policy_old.load_state_dict(torch.load('/wisnunugroho21/My Drive/cartpole/old_actor_pong_ppo_rnd.pth'))  
        
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
    
def main():
    ############## Hyperparameters ##############
    using_google_drive = True # If you using Google Colab and want to save the model to your GDrive, set this to True
    load_weights = False # If you want to load the model, set this to True
    save_weights = True # If you want to save the model, set this to True
    training_mode = True # If you want to train the model, set this to True. But set this otherwise if you only want to test it
    
    render = False # If you want to display the image. Turn this off if you run this in Google Collab
    n_update = 1 # How many episode before you update the Policy
    n_plot_batch = 100 # How many episode you want to plot the result
    n_rnd_update = 128 # How many episode before you update the RND
    n_episode = 3000 # How many episode you want to run
    #############################################         
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
        
    utils = Utils()     
    ppo = Agent(state_dim, action_dim)  
    ############################################# 
    
    if using_google_drive:
        drive.mount('/wisnunugroho21')
    
    if load_weights:
        ppo.load_weights()
        print('Weight Loaded')
    else :
        ppo.lets_init_weights()
        print('Init Weight')
    
    if torch.cuda.is_available() :
        print('Using GPU')
    
    rewards = []   
    batch_rewards = []
    
    times = []
    batch_times = []
    
    t_rnd = 0
    
    for i_episode in range(1, n_episode):
        ############################################
        state = env.reset()           
        done = False
        total_reward = 0
        t = 0
        ############################################
        
        while not done:
            # Running policy_old:            
            action = int(ppo.act(state))
            state_n, reward, done, info = env.step(action)
            
            t += 1
            t_rnd += 1                           
            total_reward += reward    
            
            # There is a bugs in cartpole env which always giving same rewards, even if the state will make env terminate (done = true)
            # So, i tweak the reward system
            # reward will negative if the state will make env terminate (done = true)
            reward = -100 if done and t < 200 else 1
              
            if training_mode:
                ppo.save_eps(state, reward, state_n, done) 
                ppo.save_observation(state) 
                
            state = state_n     
            
            if training_mode:
                if t_rnd == n_rnd_update:
                    ppo.update_rnd()
                    #print('RND has been updated')
                    t_rnd = 0
            
            if render:
                env.render('rgb_array')
            if done:
                print('Episode {} t_reward: {} time: {} '.format(i_episode, total_reward, t))
                batch_rewards.append(total_reward)
                batch_times.append(t)
                break        
        
        if training_mode:
            # update after n episodes
            if i_episode % n_update == 0 and i_episode != 0:
                ppo.update()
                #print('Agent has been updated')

                if save_weights:
                    ppo.save_weights()
                    #print('Weights saved')
            
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
            
    print('========== Final ==========')
     # Plot the reward, times for every episode
    plot(rewards)
    plot(times) 
            
if __name__ == '__main__':
    main()