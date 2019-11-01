import gym
from gym.envs.registration import register
    
import torch
import torch.nn as nn
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np

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
        self.value_layer = nn.Sequential(
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
        self.value_layer.apply(self.init_weights)
        
    def init_weights(self, m):
        for name, param in m.named_parameters():
            if 'bias' in name:
               nn.init.constant_(param, 0.01)
            elif 'weight' in name:
                nn.init.kaiming_uniform_(param, mode = 'fan_in', nonlinearity = 'relu')
        
    def forward(self, state, is_act = False):
        if is_act: 
            return self.actor_layer(state)
        else:
            return self.actor_layer(state), self.value_layer(state)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.dones = []     
        self.next_states = []
        
    def save_eps(self, state, reward, next_states, done):
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
      
    def normalize(self, data):
        data_normalized = (data - torch.mean(data)) / torch.std(data)
        return data_normalized
      
    def to_numpy(self, datas):
        if torch.cuda.is_available():
            datas = datas.cpu().detach().numpy()
        else:
            datas = datas.detach().numpy()            
        return datas        
      
    def monte_carlo_discounted(self, datas):
        # Discounting future reward        
        returns = []        
        running_add = 0
        
        for i in reversed(range(len(datas))):
            running_add = running_add * self.gamma + datas[i]
            returns.insert(0, running_add)
            
        return torch.stack(returns)
      
    def temporal_difference(self, rewards, next_values, dones):
        # Computing temporal difference
        TD = rewards + self.gamma * next_values * (1 - dones)        
        return TD
      
    def generalized_advantage_estimation(self, values, rewards, next_value, done):
        # Computing general advantages estimator
        gae = 0
        returns = []
        
        for step in reversed(range(len(rewards))):   
            delta = rewards[step] + self.gamma * next_value[step] * (1 - done[step]) - values[step]
            gae = delta + self.gamma * self.lam * gae
            returns.insert(0, gae)
            
        return torch.stack(returns)
        
class Agent:  
    def __init__(self, state_dim, action_dim):        
        self.policy_clip = 0.1 
        self.value_clip = 0.1      
        self.entropy_coef = 0.01
        self.vf_loss_coef = 0.5

        self.PPO_epochs = 5
        
        self.policy = PPO_Model(state_dim, action_dim)
        self.policy_old = PPO_Model(state_dim, action_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = 0.001)

        self.memory = Memory()
        self.utils = Utils()        
        
    def save_eps(self, state, reward, next_states, done):
        self.memory.save_eps(state, reward, next_states, done)
        
    def save_observation(self, obs):
        self.memory.save_observation(obs)

    # Loss for PPO
    def get_loss(self, states, actions, rewards, next_states, dones):      
        action_probs, values  = self.policy(states)  
        old_action_probs, old_values = self.policy_old(states)
        _, next_values  = self.policy(next_states)
        
        # Don't update old value
        old_values = old_values.detach()
                
        # Getting entropy from the action probability
        dist_entropy = self.utils.entropy(action_probs).mean()

        # Getting external general advantages estimator
        advantages = self.utils.generalized_advantage_estimation(values, rewards, next_values, dones).detach()
        returns = self.utils.temporal_difference(rewards).detach()
        
        # Getting External critic loss by using Clipped critic value
        vpredclipped = old_values + torch.clamp(values - old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        vf_losses1 = (returns - values).pow(2) # Mean Squared Error
        vf_losses2 = (returns - vpredclipped).pow(2) # Mean Squared Error
        critic_loss = torch.max(vf_losses1, vf_losses2).mean() * 0.5

        # Finding the ratio (pi_theta / pi_theta__old):  
        logprobs = self.utils.logprob(action_probs, actions) 
        old_logprobs = self.utils.logprob(old_action_probs, actions).detach()
        
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
        state = torch.FloatTensor(state).to(device)      
        action_probs = self.policy_old(state, is_act = True)
        
        # Sample the action
        action = self.utils.sample(action_probs)
        
        self.memory.save_actions(action)         
        return action.item()
        
    # Update the PPO part (the actor and value)
    def update_ppo(self):        
        length = len(self.memory.states)

        # Convert list in tensor
        old_states = torch.FloatTensor(self.memory.states).to(device).detach()
        old_actions = torch.FloatTensor(self.memory.actions).to(device).detach()
        old_next_states = torch.FloatTensor(self.memory.next_states).to(device).detach()
        dones = torch.FloatTensor(self.memory.dones).view(length, 1).view(length, 1).to(device).detach() 
        rewards = torch.FloatTensor(self.memory.rewards).view(length, 1).view(length, 1).to(device).detach()
                
        # Optimize policy for K epochs:
        for epoch in range(self.PPO_epochs):
            loss, approx_kl = self.get_loss(old_states, old_actions, rewards, old_next_states, dones)          
            
            # If KL is bigger than threshold, stop update and continue to next episode
            if approx_kl > (1.5 * self.target_kl):
                print('KL greater than target. Stop update at epoch : ', epoch + 1)
                break
            
            self.policy_optimizer.zero_grad()
            loss.backward()                    
            self.policy_optimizer.step() 

        # Clear state, action, reward in memory    
        self.memory.clearMemory()
        
        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        
    def save_weights(self):
        torch.save(self.policy.state_dict(), '/test/Your Folder/actor_pong_ppo_rnd.pth')
        torch.save(self.policy_old.state_dict(), '/test/Your Folder/old_actor_pong_ppo_rnd.pth')
        
    def load_weights(self):
        self.policy.load_state_dict(torch.load('/test/Your Folder/actor_pong_ppo_rnd.pth'))        
        self.policy_old.load_state_dict(torch.load('/test/Your Folder/old_actor_pong_ppo_rnd.pth')) 
        
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

def run_episode(env, agent, state_dim, render, training_mode):
    utils = Utils()
    ############################################
    
    state = env.reset()
    state = utils.prepro(state)

    done = False
    total_reward = 0
    t = 0

    ############################################
    # Total reward calculation only to used for determined how good your agent to find Goals. but that value will not assign to agent.
    # More efficient the agent find the goal, more Total reward it will get
    # agent only get reward from environment.
    # I set this because i really want to know, How good my agent if it only get reward in finish line
    cell_visited = []
    total_reward = 0
    success_reward = 10
    fail_reward = 1
    visit_again_minus_reward = 0.25
    travel_minus_reward = 0.04
    ############################################
    
    while not done:
        state_val = to_categorical(state, num_classes = state_dim) # One hot encoding for state because it's more efficient for Neural Network
        action = int(agent.act(state_val))
        state_n, reward, done, info = env.step(action)
        state_n = utils.prepro(state_n)

        if reward == 0 and done :
            total_reward -= fail_reward

        elif reward == 1 and done:
            total_reward += success_reward

        elif state in cell_visited:
            total_reward -= visit_again_minus_reward

        else:
            total_reward -= travel_minus_reward  
            cell_visited.append(state)
        
        t += 1  
          
        if training_mode:
            next_state_val = to_categorical(state_n, num_classes = state_dim)  # One hot encoding for next state
            agent.save_eps(state_val, reward, next_state_val, done) 
            
        state = state_n     
                
        if render:
            env.render()
        if done:
            return total_reward, t
    
def main():
    ############## Hyperparameters ##############
    using_google_drive = True # If you using Google Colab and want to save the agent to your GDrive, set this to True
    load_weights = False # If you want to load the agent, set this to True
    save_weights = True # If you want to save the agent, set this to True
    training_mode = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    
    render = False # If you want to display the image. Turn this off if you run this in Google Collab
    n_update = 1 # How many episode before you update the Policy
    n_plot_batch = 100 # How many episode you want to plot the result
    n_episode = 5000 # How many episode you want to run
    #############################################         
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.n
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
    
    rewards = []   
    batch_rewards = []
    batch_solved_reward = []
    
    times = []
    batch_times = []
    
    for i_episode in range(1, n_episode):
        total_reward, time = run_episode(env, agent, state_dim, render, training_mode)
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, total_reward, time))
        batch_rewards.append(total_reward)
        batch_times.append(time)       
        
        if training_mode:
            # update after n episodes
            if i_episode % n_update == 0 and i_episode != 0:
                agent.update_ppo()
                print('Agent has been updated')

                if save_weights:
                    agent.save_weights()
                    print('Weights saved')
                    
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
            
if __name__ == '__main__':
    main()
