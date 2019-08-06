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
                nn.init.kaiming_normal_(param, mode = 'fan_in', nonlinearity = 'relu')
        
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
                nn.Linear(64, 10)
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
                nn.Linear(64, 10)
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
        self.policy_clip = 0.2 
        self.value_clip = 1       
        self.entropy_coef = 0.01
        self.vf_loss_coef = 1
        self.target_kl = 0.1

        self.PPO_epochs = 5
        self.RND_epochs = 4
        
        self.ex_advantages_coef = 2
        self.in_advantages_coef = 1
        
        self.policy = PPO_Model(state_dim, action_dim)
        self.policy_old = PPO_Model(state_dim, action_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr = 0.001) 

        self.rnd_predict = RND_predictor_Model(state_dim, action_dim)
        self.rnd_predict_optimizer = torch.optim.Adam(self.rnd_predict.parameters(), lr = 0.001)
        self.rnd_target = RND_target_Model(state_dim, action_dim)

        self.memory = Memory()
        self.utils = Utils()        
        
    def save_eps(self, state, reward, next_states, done):
        self.memory.save_eps(state, reward, next_states, done)
        
    def save_observation(self, obs):
        self.memory.save_observation(obs)

    # Loss for RND 
    def get_rnd_loss(self, old_states):
        state_pred = self.rnd_predict(old_states)
        state_target = self.rnd_target(old_states)
        
        # Don't update target state value
        state_target = state_target.detach()
        
        # Mean Squared Error Calculation between state and predict
        forward_loss = (state_target - state_pred).pow(2).mean()
        return forward_loss

    # Loss for PPO
    def get_loss(self, old_states, old_actions, rewards, old_next_states, dones):      
        action_probs, in_value, ex_value  = self.policy(old_states)  
        old_action_probs, in_old_value, ex_old_value = self.policy_old(old_states)
        _, next_in_value, next_ex_value = self.policy(old_next_states)
        
        state_pred = self.rnd_predict(old_states)
        state_target = self.rnd_target(old_states)
        
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
        
        # Finding Intrinsic Value Function Loss by using Clipped Rewards Value
        in_vpredclipped = in_old_value + torch.clamp(in_value - in_old_value, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        in_vf_losses1 = (intrinsic_rewards - in_value).pow(2) #Mean Squared Error
        in_vf_losses2 = (intrinsic_rewards - in_vpredclipped).pow(2) #Mean Squared Error
        critic_int_loss = torch.max(in_vf_losses1, in_vf_losses2).mean()
        
        # Finding External Value Function Loss by using Clipped Rewards Value
        ex_vpredclipped = ex_old_value + torch.clamp(ex_value - ex_old_value, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        ex_vf_losses1 = (external_rewards - ex_value).pow(2) # Mean Squared Error
        ex_vf_losses2 = (external_rewards - ex_vpredclipped).pow(2) # Mean Squared Error
        critic_ext_loss = torch.max(ex_vf_losses1, ex_vf_losses2).mean()
        
        # Getting overall critic loss
        critic_loss = critic_ext_loss + critic_int_loss

        # Finding the ratio (pi_theta / pi_theta__old):  
        logprobs = self.utils.logprob(action_probs, old_actions) 
        old_logprobs = self.utils.logprob(old_action_probs, old_actions).detach()
        
        # Finding Surrogate Loss:
        ratios = torch.exp(logprobs - old_logprobs) # ratios = old_logprobs / logprobs
        surr1 = ratios * advantages * -1
        surr2 = torch.clamp(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * advantages * -1
        pg_loss = torch.max(surr1, surr2).mean()       
        
        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss and 
        loss = pg_loss + (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) 
        
        # Approx KL to choose whether we must continue the gradient descent
        approx_kl = 0.5 * (logprobs - old_logprobs).pow(2).mean()
        
        return loss, approx_kl       
      
    def act(self, state):
        state = torch.FloatTensor(state).to(device)      
        action_probs = self.policy_old(state, is_act = True)
        
        # Sample the action
        action = self.utils.sample(action_probs)
        
        self.memory.save_actions(action)         
        return action.item() 

    # Update the RND part (the state and predictor)
    def update_rnd(self):
        # Convert list in tensor
        old_states = torch.FloatTensor(self.memory.observation).to(device).detach()
        
        # Optimize predictor for K epochs:
        for epoch in range(self.RND_epochs):                        
            loss = self.get_rnd_loss(old_states)  
            
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
        torch.save(self.policy.state_dict(), '/wisnunugroho21/My Drive/frozenlake/actor_pong_ppo_rnd.pth')
        torch.save(self.policy_old.state_dict(), '/wisnunugroho21/My Drive/frozenlake/old_actor_pong_ppo_rnd.pth')
        
    def load_weights(self):
        self.policy.load_state_dict(torch.load('/wisnunugroho21/My Drive/frozenlake/actor_pong_ppo_rnd.pth'))        
        self.policy_old.load_state_dict(torch.load('/wisnunugroho21/My Drive/frozenlake/old_actor_pong_ppo_rnd.pth'))  
        
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
    ############################################
    state = env.reset()           
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
        # Running policy_old:   
        state_val = to_categorical(state, num_classes = state_dim) # One hot encoding for state because it's more efficient for Neural Network
        action = int(agent.act(state_val))
        state_n, reward, done, _ = env.step(action)
        
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
        t_rnd += 1
          
        if training_mode:
            next_state_val = to_categorical(state_n, num_classes = state_dim)  # One hot encoding for next state   
            agent.save_eps(state_val, reward, next_state_val, done) 
            agent.save_observation(state_val) 
            
        state = state_n     
        
        if training_mode:
            if t_rnd == n_rnd_update:
                agent.update_rnd()
                #print('RND has been updated')
                t_rnd = 0
        
        if render:
            env.render()
        if done:
            return total_reward, t, int(reward), t_rnd

def main():
    try:
        register(
            id='FrozenLakeNotSlippery-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name' : '4x4', 'is_slippery': False},
            max_episode_steps=100,
            reward_threshold=0.8196, # optimum = .8196
        )

        print('Env FrozenLakeNotSlippery has not yet initialized. \nInitializing now...')
    except:
        print('Env FrozenLakeNotSlippery has been initialized')
    ############## Hyperparameters ##############
    using_google_drive = False # If you using Google Colab and want to save the agent to your GDrive, set this to True
    load_weights = False # If you want to load the agent, set this to True
    save_weights = False # If you want to save the agent, set this to True
    training_mode = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold = None # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to set this off
    
    render = False # If you want to display the image. Turn this off if you run this in Google Collab
    n_update = 1 # How many episode before you update the Policy
    n_plot_batch = 100 # How many episode you want to plot the result
    n_rnd_update = 32 # How many episode before you update the RND
    n_episode = 3000 # How many episode you want to run
    #############################################         
    env_name = "FrozenLakeNotSlippery-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
        
    utils = Utils()     
    agent = Agent(state_dim, action_dim)  
    ############################################# 
    
    if using_google_drive:
        from google.colab import drive
        drive.mount('/wisnunugroho21')
    
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
    
    reach_goal = []
    batch_reach_goal = []
    
    t_rnd = 0
    
    for i_episode in range(1, n_episode):
        total_reward, time, r_goal, t_rnd = run_episode(env, agent, state_dim, render, t_rnd, training_mode, n_rnd_update)
        print('Episode {} \t t_reward: {} \t time: {} \t is_reach_goal: {}'.format(i_episode, total_reward, time, bool(r_goal)))
        batch_rewards.append(total_reward)
        batch_times.append(time)
        batch_reach_goal.append(r_goal)      
        
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
                        
                    for rg in batch_reach_goal:
                        reach_goal.append(rg)

                    print('You solved task after {} episode'.format(len(rewards)))
                    break

                else:
                    del batch_solved_reward[0]
                    batch_solved_reward.append(total_reward)

            else:
                batch_solved_reward.append(total_reward)
            
        if i_episode % n_plot_batch == 0 and i_episode != 0:
            # Plot the reward, times, and reach_goal for every n_plot_batch
            plot(batch_rewards)
            plot(batch_times)
            plot(batch_reach_goal)
            
            for reward in batch_times:
                rewards.append(reward)
                
            for time in batch_rewards:
                times.append(time)
                
            for rg in batch_reach_goal:
                reach_goal.append(rg)
                
            batch_rewards = []
            batch_times = []
            batch_reach_goal = []
            
    print('========== Final ==========')
    # Plot the reward, times, and reach_goal for every episode
    plot(rewards)
    plot(times) 
    plot(reach_goal) 
            
if __name__ == '__main__':
    main()