import gym
from gym.envs.registration import register
    
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import to_categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
      
class Model(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Model, self).__init__()
        
        # actor
        self.actor_layer = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, action_dim),
                nn.Tanh()
              ).float().to(device)
        
        # critic
        self.value_in_layer = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
              ).float().to(device)
        
        self.value_ex_layer = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
              ).float().to(device)
        
        self.state_predict_layer = nn.Sequential(
                nn.Linear(state_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 10)
              ).float().to(device)
        
        self.state_target_layer = nn.Sequential(
                nn.Linear(state_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 16),
                nn.ReLU(),
                nn.Linear(16, 10)
              ).float().to(device)
        
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
        
    def save_eps(self, state, reward, next_states, done):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        next_states = torch.FloatTensor(next_states.reshape(1, -1)).to(device)
        
        self.rewards.append(reward)
        self.states.append(state)
        self.dones.append(done)
        self.next_states.append(next_states)
        
    def save_actions(self, action):
        self.actions.append(action)
        
    def save_logprobs(self, logprob):
        self.logprobs.append(logprob)
        
    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        
class Utils:
    def __init__(self, action_dim):
        self.gamma = 0.95
        self.diagonal = torch.full((action_dim,), 0.01).float().to(device)
        
    # MultiVariable Normal Distribution is used for continous action space
    # The distribution need mean and standard deviation
    # The neural network output the mean of distribution 
    # The standard deviation is assigned by parameter

    def sample(self, datas):   
        distribution = MultivariateNormal(datas, torch.diag(self.diagonal))   
        return distribution.sample().float().to(device)
        
    def entropy(self, datas):      
        distribution = MultivariateNormal(datas, torch.diag(self.diagonal))       
        return distribution.entropy().float().to(device)
      
    def logprob(self, datas, value_data):      
        distribution = MultivariateNormal(datas, torch.diag(self.diagonal))         
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
      
    def q_values(self, reward, next_state, done, value_function):
        # Finding Q Values
        q_values = reward + (1 - done) * self.gamma * value_function(next_state)           
        return q_values
        
class Agent:  
    def __init__(self, state_dim, action_dim):        
        self.eps_clip = 0.2
        self.K_epochs = 5
        self.entropy_coef = 0.01
        self.vf_loss_coef = 1
        self.update_proportion = 0.25
        self.target_kl = 0.1
        
        self.policy = Model(state_dim, action_dim)
        self.policy_old = Model(state_dim, action_dim)         
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters()) 
        
        self.memory = Memory()
        self.utils = Utils(action_dim)        
        
    def save_eps(self, state, reward, next_states, done):
        self.memory.save_eps(state, reward, next_states, done)
        
    def get_loss(self, old_states, old_actions, rewards):      
        action_probs, in_value, ex_value, state_pred, state_target = self.policy(old_states)  
        old_action_probs, in_old_value, ex_old_value, _, _ = self.policy_old(old_states)
        
        # Don't update old value
        old_action_probs = old_action_probs.detach()
        in_old_value = in_old_value.detach()
        ex_old_value = ex_old_value.detach()
        state_target = state_target.detach() #Don't update target state value
        
        #Reshaping back
        state_pred, state_target = state_pred.squeeze(1), state_target.squeeze(1)
        in_value, ex_value, in_old_value, ex_old_value = in_value.squeeze(1), ex_value.squeeze(1), in_old_value.squeeze(1), ex_old_value.squeeze(1)
                
        # Getting entropy from the action probability
        dist_entropy = self.utils.entropy(action_probs).mean()
        
        # Discounting external reward and getting external advantages
        external_rewards = self.utils.discounted(rewards).detach()
        external_advantage = external_rewards - ex_value
                    
        # Finding and discounting internal reward, then getting internal advantages
        intrinsic_rewards = (state_target - state_pred).pow(2).sum(1)
        intrinsic_rewards = self.utils.discounted(intrinsic_rewards).detach()
        intrinsic_advantage = intrinsic_rewards - in_value          
        
        # Getting overall advantages
        advantages = (external_advantage + intrinsic_advantage).detach()
        
        # Getting loss for state predictor
        forward_loss = (state_target - state_pred).pow(2).mean(1)        
        mask = torch.rand(len(forward_loss)) #Create random array
        mask = (mask < self.update_proportion).type(dataType) #Using random array to choose whether we must update the state predict or not    
        forward_loss = (forward_loss * mask).sum() / torch.max(mask.sum(), torch.Tensor([1]).to(device))
        
        # Finding Intrinsic Value Function Loss by using Clipped Rewards Value
        in_vpredclipped = in_old_value + torch.clamp(in_value - in_old_value, -self.eps_clip, self.eps_clip) # Minimize the difference between old value and new value
        in_vf_losses1 = (intrinsic_rewards - in_value).pow(2) #Mean Squared Error
        in_vf_losses2 = (intrinsic_rewards - in_vpredclipped).pow(2) #Mean Squared Error
        critic_int_loss = torch.min(in_vf_losses1, in_vf_losses2).mean()
        
        # Finding External Value Function Loss by using Clipped Rewards Value
        ex_vpredclipped = ex_old_value + torch.clamp(ex_value - ex_old_value, -self.eps_clip, self.eps_clip) # Minimize the difference between old value and new value
        ex_vf_losses1 = (external_rewards - ex_value).pow(2) #Mean Squared Error
        ex_vf_losses2 = (external_rewards - ex_vpredclipped).pow(2) #Mean Squared Error
        critic_ext_loss = torch.min(ex_vf_losses1, ex_vf_losses2).mean()
        
        # Getting overall critic loss
        critic_loss = critic_ext_loss + critic_int_loss

        # Finding the logprobs and old_logprobs:    
        logprobs = self.utils.logprob(action_probs, old_actions) 
        old_logprobs = self.utils.logprob(old_action_probs, old_actions).detach()
        
        # Finding Surrogate Loss for actor:
        ratios = torch.exp(logprobs - old_logprobs) # ratios = old_logprobs / logprobs
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        pg_loss = torch.min(surr1, surr2).mean()        
        
        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss and Forward Loss to sharpen their prediction skill
        loss = pg_loss - (critic_loss * self.vf_loss_coef) + (dist_entropy * self.entropy_coef) - forward_loss 
        loss = loss * -1
        
        approx_kl = 0.5 * (logprobs - old_logprobs).pow(2).mean()
        
        return loss, approx_kl     
      
    def act(self, state):
        state = torch.FloatTensor(state).to(device)      
        action_probs, _, _, _, _ = self.policy_old(state)
        
        # Sample the action
        action = self.utils.sample(action_probs)        
        self.memory.save_actions(action)   
        
        return action.item() 
        
    def update(self):        
        # Convert list in tensor
        old_states = torch.stack(self.memory.states).to(device).detach()
        old_actions = torch.stack(self.memory.actions).to(device).detach()
        rewards = torch.FloatTensor(self.memory.rewards).to(device).detach()
                
        # Optimize policy for K epochs:
        for epoch in range(self.K_epochs):            
            loss, approx_kl = self.get_loss(old_states, old_actions, rewards)            
            
            if approx_kl > (1.5 * self.target_kl):
                print('KL greater than target. Stop update at epoch : ', epoch)
                break
            
            self.policy_optimizer.zero_grad()
            loss.backward()                    
            self.policy_optimizer.step() 
            
        self.memory.clearMemory()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
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
    env_name = "Pendulum-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
        
    render = False
    n_update = 1
    #############################################    
            
    ppo = Agent(state_dim, action_dim) 
    
    if torch.cuda.is_available() :
        print('Using GPU')
    
    rewards = []   
    batch_rewards = []
    
    times = []
    batch_times = []
    
    
    for i_episode in range(1, 100000):
        ############################################
        state = env.reset()
        done = False
        total_reward = 0
        t = 0
        ############################################
        
        while not done:
            # Running policy_old:   
            action = ppo.act(state) 
            action *= 2
            action = np.array([action])
            
            state_n, reward, done, _ = env.step(action)
            
            total_reward += reward
            t += 1
             
            ppo.save_eps(state, reward, state_n, done) 
            state = state_n       
            
            if render:
                env.render()
            if done:
                print('Episode {} t_reward: {} time: {}'.format(i_episode, total_reward, t))
                batch_rewards.append(total_reward)
                batch_times.append(t)
                break        
        
        # update after n episodes
        if i_episode % n_update == 0 and i_episode != 0:
            ppo.update()
            
        if i_episode % 100 == 0 and i_episode != 0:
            plot(batch_rewards)
            plot(batch_times)
            
            for reward in batch_times:
                rewards.append(reward)
                
            for time in batch_rewards:
                times.append(time)
                
            batch_rewards = []
            batch_times = []
            
    print('========== Final ==========')
    plot(rewards)
    plot(times)    
            
if __name__ == '__main__':
    main()