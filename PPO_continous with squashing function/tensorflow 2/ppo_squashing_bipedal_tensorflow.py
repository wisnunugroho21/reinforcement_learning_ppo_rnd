import gym
from gym.envs.registration import register
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

class Actor_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Actor_Model, self).__init__()

        self.d1 = Dense(64, activation='relu', name = 'a1')
        self.d2 = Dense(64, activation='relu', name = 'a2')

        self.out_mean = Dense(action_dim, activation='linear', name = 'a3')
        self.out_logstd = Dense(action_dim, activation='linear', name = 'a4')

    def call(self, states):
        x = self.d1(states)
        x = self.d2(x)

        mean    = self.out_mean(x)
        log_std = self.out_logstd(x)
        log_std = tf.clip_by_value(log_std, -20, 2) # Making sure log_std is not too far

        return mean, log_std

class Critic_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic_Model, self).__init__()

        self.d1 = Dense(64, activation='relu', name = 'c1')
        self.d2 = Dense(64, activation='relu', name = 'c2')
        self.out = Dense(1, activation='linear', name = 'c4')

    def call(self, states):
        x = self.d1(states)
        x = self.d2(x)
        return self.out(x)

class Memory:
    def __init__(self):
        self.z = [] # Z is action before squashed by tanh
        self.states = []
        self.rewards = []
        self.dones = []     
        self.next_states = []        

    def save_eps(self, state, reward, z, done, next_state):
        self.rewards.append(reward)
        self.states.append(state.tolist())
        self.z.append(z)
        self.dones.append(float(done))
        self.next_states.append(next_state.tolist())        

    def clearMemory(self):
        del self.z[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]        

    def length(self):
        return len(self.dones)

    def get_all_items(self):         
        states = tf.constant(self.states, dtype = tf.float32)
        z = tf.constant(self.z, dtype = tf.float32) # Z is action before squashed by tanh
        rewards = tf.expand_dims(tf.constant(self.rewards, dtype = tf.float32), 1)
        dones = tf.expand_dims(tf.constant(self.dones, dtype = tf.float32), 1)
        next_states = tf.constant(self.next_states, dtype = tf.float32)        

        return tf.data.Dataset.from_tensor_slices((states, z, rewards, dones, next_states)) 

class Utils:
    def __init__(self):
        self.gamma = 0.99
        self.lam = 0.95

    # If you want to create Agent for Continous Action Environment, you must find the proper Distribution for it (Some people use Multivariate Gaussian Distribution)
    # and making the neural network output directly the action, not probability (Deterministic policy).

    # std is not static number, but a number outputed by Policy Neural Network. 

    # This link may will help you : https://www.reddit.com/r/learnmachinelearning/comments/8gp7ze/rl_in_continuous_action_space_how_to_predict_the/

    def sample(self, mean, std):
        distribution = tfp.distributions.Normal(mean, std)
        return distribution.sample()

    def entropy(self, mean, std):
        distribution = tfp.distributions.Normal(mean, std)            
        return distribution.entropy()

    def logprob(self, mean, std, value_data):
        distribution = tfp.distributions.Normal(mean, std)
        return distribution.log_prob(value_data) - tf.math.log(1 - tf.math.square(tf.math.tanh(value_data)) + 1e-6) # The logprob must be modified because squashing function

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

    def generalized_advantage_estimation(self, values, rewards, next_value, done):
        # Computing general advantages estimator
        gae = 0
        returns = []

        for step in reversed(range(len(rewards))):   
            delta = rewards[step] + self.gamma * next_value[step] * (1 - done[step]) - values[step]
            gae = delta + (self.lam * gae)
            returns.insert(0, gae)

        return tf.stack(returns)

class Agent:  
    def __init__(self, state_dim, action_dim, is_training_mode):        
        self.policy_clip = 0.2 
        self.value_clip = 0.2    
        self.entropy_coef = 1
        self.vf_loss_coef = 1
        self.minibatch = 32        
        self.PPO_epochs = 5
        self.is_training_mode = is_training_mode

        self.actor = Actor_Model(state_dim, action_dim)
        self.actor_old = Actor_Model(state_dim, action_dim)

        self.critic = Critic_Model(state_dim, action_dim)
        self.critic_old = Critic_Model(state_dim, action_dim)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4)
        self.memory = Memory()
        self.utils = Utils()        

    def save_eps(self, state, reward, z, done, next_state):
        self.memory.save_eps(state, reward, z, done, next_state)

    # Loss for PPO  
    def get_loss(self, states, z, rewards, dones, next_states):        
        action_mean, action_logstd = self.actor(states)
        values = self.critic(states)
        old_action_mean, old_action_logstd = self.actor_old(states)
        old_values = self.critic_old(states)
        next_values  = self.critic(next_states)

        action_std = tf.math.exp(action_logstd) # The output of Policy NN is log_std. Therefore, we must convert it to std via exp function
        old_action_std = tf.math.exp(old_action_logstd) 

        # Don't use old value in backpropagation
        Old_values = tf.stop_gradient(old_values)

        # Getting external general advantages estimator
        Advantages = tf.stop_gradient(self.utils.generalized_advantage_estimation(values, rewards, next_values, dones))
        Returns = tf.stop_gradient(self.utils.temporal_difference(rewards, next_values, dones))

        # Finding the ratio (pi_theta / pi_theta__old):        
        logprobs = self.utils.logprob(action_mean, action_std, z)
        Old_logprobs = tf.stop_gradient(self.utils.logprob(old_action_mean, old_action_std, z))

        # Getting entropy from the action probability 
        dist_entropy = tf.math.reduce_mean(self.utils.entropy(action_mean, action_std))

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
        # and minimize Critic Loss 
        loss = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss       

    @tf.function
    def act(self, state):        
        state = tf.expand_dims(tf.cast(state, dtype = tf.float32), 0)        
        action_mean, action_logstd = self.actor(state)
        action_std = tf.math.exp(action_logstd)

        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions        
        if self.is_training_mode:
            # Sample the action
            z = self.utils.sample(action_mean, action_std)
            action = tf.math.tanh(z) # Squashing by tanh will make action bound in range -1 to 1
        else:
            action = action_mean
            entropy = 0

        return tf.squeeze(action), tf.squeeze(z)

    # Get loss and Do backpropagation
    @tf.function
    def training_ppo(self, states, z, rewards, dones, next_states):        
        with tf.GradientTape() as tape:
            loss = self.get_loss(states, z, rewards, dones, next_states)

        gradients = tape.gradient(loss, self.actor.trainable_variables + self.critic.trainable_variables)        
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic.trainable_variables)) 

    # Update the model
    def update_ppo(self):        
        batch_size = int(self.memory.length() / self.minibatch)

        # Optimize policy for K epochs:
        for epoch in range(self.PPO_epochs):       
            for states, z, rewards, dones, next_states in self.memory.get_all_items().batch(batch_size):
                self.training_ppo(states, z, rewards, dones, next_states)

        # Clear the memory
        self.memory.clearMemory()

        # Copy new weights into old policy:
        self.actor_old.set_weights(self.actor.get_weights())
        self.critic_old.set_weights(self.critic.get_weights())

    def save_weights(self):
        self.actor.save_weights('bipedal_weights/actor_pong_ppo_rnd.hd5')
        self.actor_old.save_weights('bipedal_weights/actor_old_pong_ppo_rnd.hd5')
        self.critic.save_weights('bipedal_weights/critic_pong_ppo_rnd.hd5')
        self.critic_old.save_weights('bipedal_weights/critic_old_pong_ppo_rnd.hd5')

    def load_weights(self):
        self.actor.load_weights('bipedal_weights/actor_pong_ppo_rnd.hd5')
        self.actor_old.load_weights('bipedal_weights/actor_old_pong_ppo_rnd.hd5')
        self.critic.load_weights('bipedal_weights/critic_pong_ppo_rnd.hd5')
        self.critic_old.load_weights('bipedal_weights/critic_old_pong_ppo_rnd.hd5')

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
    state = env.reset()     
    done = False
    total_reward = 0
    eps_time = 0
    ############################################

    while not done:
        action, z = agent.act(state)
        z = z.numpy()         
        next_state, reward, done, info = env.step(action)

        eps_time += 1 
        t_updates += 1
        total_reward += reward

        if training_mode:
            agent.save_eps(state, reward, z, done, next_state)

        state = next_state     

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
    load_weights = False # If you want to load the agent, set this to True
    save_weights = False # If you want to save the agent, set this to True
    training_mode = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold = 200 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off

    render = False # If you want to display the image. Turn this off if you run this in Google Collab
    n_update = 2048 # How many episode before you update the Policy
    n_plot_batch = 100 # How many episode you want to plot the result
    n_episode = 10000 # How many episode you want to run
    ############################################# 
    env_name = 'BipedalWalker-v2'
    env = gym.make(env_name)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

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
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, total_reward, time))
        batch_rewards.append(int(total_reward))
        batch_times.append(time)        

        if save_weights:
            agent.save_weights()  
            print('weights saved')

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