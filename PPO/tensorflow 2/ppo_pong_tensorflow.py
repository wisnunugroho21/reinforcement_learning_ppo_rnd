import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

class Actor_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Actor_Model, self).__init__()
        
        self.layer_actor_1 = Dense(640, activation='relu', name = 'a1')
        self.layer_actor_2 = Dense(160, activation='relu', name = 'a2')
        self.layer_actor_out = Dense(action_dim, activation='softmax', name = 'a3')
        
    def call(self, states):
        x = self.layer_actor_1(states)
        x = self.layer_actor_2(x)
        return self.layer_actor_out(x)
          
class Critic_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic_Model, self).__init__()
        
        self.layer_critic_1 = Dense(640, activation='relu', name = 'c1')
        self.layer_critic_2 = Dense(160, activation='relu', name = 'c2')
        self.layer_critic_out = Dense(1, activation='linear', name = 'c3')
        
    def call(self, states):
        x = self.layer_critic_1(states)
        x = self.layer_critic_2(x)
        return self.layer_critic_out(x)
      
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.rewards = []
        self.dones = []     
        self.next_states = []
        
    def save_eps(self, state, reward, action, done, next_state):
        self.rewards.append(reward)
        self.states.append(state.tolist())
        self.actions.append(action)
        self.dones.append(float(done))
        self.next_states.append(next_state.tolist())
                
    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        
    def length(self):
        return len(self.actions)
        
    def get_all_items(self):        
        states = tf.constant(self.states, dtype = tf.float32)
        actions = tf.constant(self.actions, dtype = tf.float32)
        rewards = tf.expand_dims(tf.constant(self.rewards, dtype = tf.float32), 1)
        dones = tf.expand_dims(tf.constant(self.dones, dtype = tf.float32), 1)
        next_states = tf.constant(self.next_states, dtype = tf.float32)
        
        return tf.data.Dataset.from_tensor_slices((states, actions, rewards, dones, next_states)) 
        
class Utils:
    def __init__(self):
        self.gamma = 0.99
        self.lam = 0.95

    # Categorical Distribution is used for Discrete Action Environment
    # The neural network output the probability of actions (Stochastic policy), then pass it to Categorical Distribution
    
    def sample(self, datas):
        distribution = tfp.distributions.Categorical(probs = datas)
        return distribution.sample()
        
    def entropy(self, datas):
        distribution = tfp.distributions.Categorical(probs = datas)            
        return distribution.entropy()
      
    def logprob(self, datas, value_data):
        distribution = tfp.distributions.Categorical(probs = datas)
        return distribution.log_prob(value_data)
      
    def normalize(self, data):
        data_normalized = (data - tf.math.reduce_mean(data)) / (tf.math.reduce_std(data) + 1e-8)
        return data_normalized     
      
    def discounted(self, datas, dones):
        # Discounting future reward        
        returns = []        
        running_add = 0
        
        for i in reversed(range(len(datas))):
            running_add = datas[i] + self.gamma * running_add * (1 - dones)
            returns.insert(0, running_add)
            
        return tf.stack(returns)
      
    def temporal_difference(self, rewards, next_values, dones):
        # Finding TD Values
        # TD = R + V(St+1)
        TD = rewards + self.gamma * next_values * (1 - dones)        
        return TD
      
    def compute_GAE(self, values, rewards, next_value, done):
        # Computing general advantages estimator
        gae = 0
        returns = []
        
        for step in reversed(range(len(rewards))):   
            delta = rewards[step] + self.gamma * next_value[step] * (1 - done[step]) - values[step]
            gae = delta + (self.lam * gae)
            returns.insert(0, gae)
            
        return tf.stack(returns)
      
    def prepro(self, I):
        I = I[35:195] # crop
        I = I[::2,::2, 0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0] = 1 # everything else (paddles, ball) just set to 1
        
        X = I.astype(np.float32).ravel() # Combine items in 1 array 
        return X
      
class Agent:  
    def __init__(self, state_dim, action_dim, is_training_mode):        
        self.policy_clip = 0.1 
        self.value_clip = 0.1    
        self.entropy_coef = 0.01
        self.vf_loss_coef = 1
        self.minibatch = 4        
        self.PPO_epochs = 4

        self.is_training_mode = is_training_mode
                
        self.actor = Actor_Model(state_dim, action_dim)
        self.actor_old = Actor_Model(state_dim, action_dim)
        
        self.critic = Critic_Model(state_dim, action_dim)
        self.critic_old = Critic_Model(state_dim, action_dim)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 2.5e-4)
        self.memory = Memory()
        self.utils = Utils()        
        
    def save_eps(self, state, reward, action, done, next_state):
        self.memory.save_eps(state, reward, action, done, next_state)

    # Loss for PPO  
    def get_loss(self, states, actions, rewards, dones, next_states):        
        action_probs, values  = self.actor(states), self.critic(states)
        old_action_probs, old_values = self.actor_old(states), self.critic_old(states)
        next_values  = self.critic(next_states)  
                
        # Don't update old value
        Old_values = tf.stop_gradient(old_values)
        
        # Getting external general advantages estimator
        Advantages = tf.stop_gradient(self.utils.compute_GAE(values, rewards, next_values, dones))
        Returns = tf.stop_gradient(self.utils.temporal_difference(rewards, next_values, dones))
        
        # Finding the ratio (pi_theta / pi_theta__old):        
        logprobs = tf.expand_dims(self.utils.logprob(action_probs, actions), 1)         
        Old_logprobs = tf.stop_gradient(tf.expand_dims(self.utils.logprob(old_action_probs, actions), 1))
        
        # Getting entropy from the action probability 
        dist_entropy = tf.math.reduce_mean(self.utils.entropy(action_probs))
                        
        # Getting External critic loss by using Clipped critic value
        vpredclipped = old_values + tf.clip_by_value(values - Old_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        vf_losses1 = tf.math.square(Returns - values) # Mean Squared Error
        vf_losses2 = tf.math.square(Returns - vpredclipped) # Mean Squared Error        
        critic_loss = tf.math.reduce_mean(tf.math.maximum(vf_losses1, vf_losses2)) * 0.5        
        
        # Finding Surrogate Loss
        ratios = tf.math.exp(logprobs - Old_logprobs) # ratios = old_logprobs / logprobs        
        surr1 = ratios * Advantages        
                
        surr2 = tf.clip_by_value(ratios, 1 - self.policy_clip, 1 + self.policy_clip) * Advantages
        pg_loss = tf.math.reduce_mean(tf.math.minimum(surr1, surr2))         
                        
        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss and 
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss       
      
    @tf.function
    def act(self, state):
        state = tf.expand_dims(tf.cast(state, dtype = tf.float32), 0)
        action_probs = self.actor(state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = self.utils.sample(action_probs)
        else:
            action = tf.math.argmax(action_probs)         
        return action    
    
    @tf.function
    def training_ppo(self, states, actions, rewards, dones, next_states):        
        with tf.GradientTape() as tape:
            loss = self.get_loss(states, actions, rewards, dones, next_states)
                    
        gradients = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables) 
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables)) 
        
    # Update the PPO part (the actor and value)
    def update_ppo(self):        
        batch_size = int(self.memory.length() / self.minibatch)
        
        # Optimize policy for K epochs:
        for epoch in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in self.memory.get_all_items().batch(batch_size):
                self.training_ppo(states, actions, rewards, dones, next_states)
                    
        # Delete memory
        self.memory.clearMemory()
                
        # Copy new weights into old policy:
        self.actor_old.set_weights(self.actor.get_weights())
        self.critic_old.set_weights(self.critic.get_weights())
        
    def save_weights(self):
        self.actor.save_weights('pong_weights/actor_pong_ppo')
        self.actor_old.save_weights('pong_weights/actor_old_pong_ppo')
        self.critic.save_weights('pong_weights/critic_pong_ppo')
        self.critic_old.save_weights('pong_weights/critic_old_pong_ppo')
        
    def load_weights(self):
        self.actor.load_weights('pong_weights/actor_pong_ppo')
        self.actor_old.load_weights('pong_weights/actor_old_pong_ppo')
        self.critic.load_weights('pong_weights/critic_pong_ppo')
        self.critic_old.load_weights('pong_weights/critic_old_pong_ppo')
        
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

def run_episode(env, agent, state_dim, render, training_mode, t_updates, n_update):
    utils = Utils()
    ############################################
    obs = env.reset()
    obs = utils.prepro(obs)
    state = obs
    
    done = False
    total_reward = 0
    eps_time = 0
    ############################################
    
    while not done:
        action = int(agent.act(state))  
        action_gym = action + 2 if action != 0 else 0
        
        next_obs, reward, done, info = env.step(action_gym)
        next_obs = utils.prepro(next_obs)
        next_state = next_obs - obs
        
        eps_time += 1 
        t_updates += 1
        total_reward += reward
          
        if training_mode:
            agent.save_eps(state, reward, action, done, next_state) 
            
        state = next_state   
        obs = next_obs
                
        if render:
            env.render()
        
        if training_mode:
            if t_updates % n_update == 0:
                agent.update_ppo()
                t_updates = 0
        
        if done:                      
            return total_reward, eps_time, t_updates
    
def main():
    ############## Hyperparameters ##############
    using_google_drive = False # If you using Google Colab and want to save the agent to your GDrive, set this to True
    load_weights = True # If you want to load the agent, set this to True
    save_weights = False # If you want to save the agent, set this to True
    training_mode = False # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold = 20 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    
    render = True # If you want to display the image. Turn this off if you run this in Google Collab
    n_update = 128 # How many episode before you update the Policy
    n_plot_batch = 10 # How many episode you want to plot the result
    n_episode = 20 # How many episode you want to run
    #############################################         
    env_name = 'Pong-v4'
    env = gym.make(env_name)
    print(env.unwrapped.get_action_meanings())
    
    state_dim = 80 * 80
    action_dim = 3
        
    utils = Utils()     
    agent = Agent(state_dim, action_dim, training_mode)  
    #############################################  
    
    if using_google_drive:
        from google.colab import drive
        drive.mount('/test')
        
    if load_weights:
        agent.load_weights()
        print('Weight Loaded')
    
    rewards = []   
    batch_rewards = []
    batch_solved_reward = []
    
    times = []
    batch_times = []

    total_time = 0
    t_updates = 0
    
    for i_episode in range(1, n_episode + 1):
        total_reward, time, t_updates = run_episode(env, agent, state_dim, render, training_mode, t_updates, n_update)
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, int(total_reward), time))
        batch_rewards.append(int(total_reward))
        batch_times.append(time)   
                            
        if reward_threshold:
            if len(batch_solved_reward) == 100:            
                if np.mean(batch_solved_reward) >= reward_threshold :              
                    for reward in batch_rewards:
                        rewards.append(reward)

                    for time in batch_times:
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
            
            if save_weights:
                agent.save_weights()  
                print('weights saved')
            
            plot(batch_rewards)
            plot(batch_times)
            
            for reward in batch_rewards:
                rewards.append(reward)
                
            for time in batch_times:
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