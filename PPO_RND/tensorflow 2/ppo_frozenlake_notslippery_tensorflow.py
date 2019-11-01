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
        
        self.layer_actor_1 = Dense(64, activation='relu', name = 'a1')
        self.layer_actor_2 = Dense(64, activation='relu', name = 'a2')
        self.layer_actor_out = Dense(action_dim, activation='softmax', name = 'a3')
        
    def call(self, states):
        ax1 = self.layer_actor_1(states)
        ax2 = self.layer_actor_2(ax1)
        return self.layer_actor_out(ax2)
          
class Critic_In_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic_In_Model, self).__init__()
        
        self.layer_critic_1 = Dense(64, activation='relu', name = 'c1')
        self.layer_critic_2 = Dense(64, activation='relu', name = 'c2')
        self.layer_critic_out = Dense(1, activation='linear', name = 'c3')
        
    def call(self, states):
        cx1 = self.layer_critic_1(states)
        cx2 = self.layer_critic_2(cx1)
        return self.layer_critic_out(cx2)
      
class Critic_Ex_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(Critic_Ex_Model, self).__init__()
        
        self.layer_critic_1 = Dense(64, activation='relu', name = 'c1')
        self.layer_critic_2 = Dense(64, activation='relu', name = 'c2')
        self.layer_critic_out = Dense(1, activation='linear', name = 'c3')
        
    def call(self, states):
        cx1 = self.layer_critic_1(states)
        cx2 = self.layer_critic_2(cx1)
        return self.layer_critic_out(cx2)
      
class RND_Predictor_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(RND_Predictor_Model, self).__init__()
        
        self.layer_critic_1 = Dense(64, activation='relu', name = 'c1')
        self.layer_critic_2 = Dense(64, activation='relu', name = 'c2')
        self.layer_critic_out = Dense(1, activation='linear', name = 'c3')
        
    def call(self, states):
        cx1 = self.layer_critic_1(states)
        cx2 = self.layer_critic_2(cx1)
        return self.layer_critic_out(cx2)
      
class RND_Target_Model(Model):
    def __init__(self, state_dim, action_dim):
        super(RND_Target_Model, self).__init__()
        
        self.layer_critic_1 = Dense(64, activation='relu', name = 'c1')
        self.layer_critic_2 = Dense(64, activation='relu', name = 'c2')
        self.layer_critic_out = Dense(1, activation='linear', name = 'c3')
        
    def call(self, states):
        cx1 = self.layer_critic_1(states)
        cx2 = self.layer_critic_2(cx1)
        return self.layer_critic_out(cx2)
      
class Memory:
    def __init__(self, state_dim, action_dim):
        self.actions = []
        self.states = []
        self.rewards = []
        self.dones = []     
        self.next_states = []
        self.observation = []
        self.mean_obs = tf.zeros(state_dim, dtype = tf.float32)
        self.std_obs = tf.zeros(state_dim, dtype = tf.float32)
        self.std_in_rewards = tf.constant([0], dtype = tf.float32)
        self.total_number_obs = tf.constant([0], dtype = tf.float32)
        self.total_number_rwd = tf.constant([0], dtype = tf.float32)
        
    def save_eps(self, state, reward, action, done, next_state):
        self.rewards.append(reward)
        self.states.append(state.tolist())
        self.actions.append(action)
        self.dones.append(float(done))
        self.next_states.append(next_state.tolist())
        
    def save_observation(self, obs):
        self.observation.append(obs.tolist())
        
    def save_observation_normalize_parameter(self, mean_obs, std_obs, total_number_obs):
        self.mean_obs = mean_obs
        self.std_obs = std_obs
        self.total_number_obs = total_number_obs
        
    def save_rewards_normalize_parameter(self, std_in_rewards, total_number_rwd):
        self.std_in_rewards = std_in_rewards
        self.total_number_rwd = total_number_rwd
        
    def clearObs(self):
        del self.observation[:]
                
    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]
        
    def length(self):
        return len(self.actions)
      
    def length_obs(self):
        return len(self.observation)
        
    def get_all_items(self):        
        states = tf.constant(self.states, dtype = tf.float32)
        actions = tf.constant(self.actions, dtype = tf.float32)
        rewards = tf.expand_dims(tf.constant(self.rewards, dtype = tf.float32), 1)
        dones = tf.expand_dims(tf.constant(self.dones, dtype = tf.float32), 1)
        next_states = tf.constant(self.next_states, dtype = tf.float32)
        
        return tf.data.Dataset.from_tensor_slices((states, actions, rewards, dones, next_states)) 
      
    def get_all_obs(self):        
        obs = tf.constant(self.observation, dtype = tf.float32)        
        return tf.data.Dataset.from_tensor_slices((obs))    
        
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
      
    def normalize(self, data, mean = None, std = None, clip = None):
        if isinstance(mean, tf.Tensor) and isinstance(std, tf.Tensor):
            data_normalized = (data - mean) / (std + 1e-8)            
        else:
            data_normalized = (data - tf.math.reduce_mean(data)) / (tf.math.reduce_std(data) + 1e-8)
                    
        if clip :
            data_normalized = tf.clip_by_value(data_normalized, -clip, clip)

        return data_normalized     
      
    def monte_carlo_discounted(self, datas, dones):
        # Discounting future reward        
        returns = []        
        running_add = 0
        
        for i in reversed(range(len(datas))):
            running_add = datas[i] + self.gamma * running_add * (1 - dones)
            returns.insert(0, running_add)
            
        return tf.stack(returns)
      
    def temporal_difference(self, rewards, next_values, dones):
        # Computing temporal difference
        TD = rewards + self.gamma * next_values * (1 - dones)        
        return TD
      
    def generalized_advantage_estimation(self, values, rewards, next_value, done):
        # Computing generalized advantages estimation
        gae = 0
        returns = []
        
        for step in reversed(range(len(rewards))):   
            delta = rewards[step] + self.gamma * next_value[step] * (1 - done[step]) - values[step]
            gae = delta + (self.lam * gae)
            returns.insert(0, gae)
            
        return tf.stack(returns)
      
    def updateNewMean(self, prevMean, prevLen, newData):
        return ((prevMean * prevLen) + tf.math.reduce_sum(newData, 0)) / (prevLen + newData.shape[0])
      
    def updateNewStd(self, prevStd, prevLen, newData):
        return tf.math.sqrt(((tf.math.square(prevStd) * prevLen) + (tf.math.reduce_variance(newData, 0) * newData.shape[0])) / (prevStd + newData.shape[0]))
      
class Agent:  
    def __init__(self, state_dim, action_dim, is_training_mode):        
        self.policy_clip = 0.1 
        self.value_clip = 1      
        self.entropy_coef = 0.001
        self.vf_loss_coef = 1
        self.minibatch = 1
        
        self.PPO_epochs = 4
        self.RND_epochs = 4
        
        self.ex_advantages_coef = 2
        self.in_advantages_coef = 1
        
        self.clip_normalization = 5
        self.is_training_mode = is_training_mode
                
        self.actor = Actor_Model(state_dim, action_dim)
        self.actor_old = Actor_Model(state_dim, action_dim)
        
        self.critic_in = Critic_In_Model(state_dim, action_dim)
        self.critic_in_old = Critic_In_Model(state_dim, action_dim)
        
        self.critic_ex = Critic_Ex_Model(state_dim, action_dim)
        self.critic_ex_old = Critic_Ex_Model(state_dim, action_dim)
        
        self.rnd_predict = RND_Predictor_Model(state_dim, action_dim)
        self.rnd_target = RND_Target_Model(state_dim, action_dim)
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 2.5e-4)
        self.rnd_optimizer = tf.keras.optimizers.Adam(learning_rate = 2.5e-4)
        
        self.memory = Memory(state_dim, action_dim)
        self.utils = Utils()        
        
    def save_eps(self, state, reward, action, done, next_state):
        self.memory.save_eps(state, reward, action, done, next_state)
        
    def save_observation(self, obs):
        self.memory.save_observation(obs)
        
    def updateObsNormalizationParam(self):
        obs = tf.constant(self.memory.observation, dtype = tf.float32)
        
        mean_obs = self.utils.updateNewMean(self.memory.mean_obs, self.memory.total_number_obs, obs)
        std_obs = self.utils.updateNewStd(self.memory.std_obs, self.memory.total_number_obs, obs)
        total_number_obs = len(obs) + self.memory.total_number_obs
        
        self.memory.save_observation_normalize_parameter(mean_obs, std_obs, total_number_obs)
        
    def updateRwdNormalizationParam(self, in_rewards):        
        std_in_rewards = self.utils.updateNewStd(self.memory.std_in_rewards, self.memory.total_number_rwd, in_rewards)
        total_number_rwd = len(in_rewards) + self.memory.total_number_rwd
        
        self.memory.save_rewards_normalize_parameter(std_in_rewards, total_number_rwd)
        
    # Loss for RND 
    def get_rnd_loss(self, obs, mean_obs, std_obs):
        obs = self.utils.normalize(obs, mean_obs, std_obs)

        state_pred = self.rnd_predict(obs)
        state_target = self.rnd_target(obs)
        
        # Don't update target state value
        state_target = tf.stop_gradient(state_target)
        
        # Mean Squared Error Calculation between state and predict
        forward_loss = tf.math.reduce_mean(tf.math.square(state_target - state_pred))
        return forward_loss, (state_target - state_pred)

    # Loss for PPO  
    def get_loss(self, std_in_rewards, mean_obs, std_obs, states, actions, rewards, dones, next_states):        
        action_probs, in_values, ex_values  = self.actor(states), self.critic_in(states), self.critic_ex(states)
        old_action_probs, in_old_values, ex_old_values  = self.actor_old(states), self.critic_in_old(states), self.critic_ex_old(states)
        next_in_values, next_ex_values  = self.critic_in(next_states), self.critic_ex(next_states) 
        
        # Normalize the observation
        obs = self.utils.normalize(next_states, mean_obs, std_obs, self.clip_normalization)
        state_pred = self.rnd_predict(obs)
        state_target = self.rnd_target(obs) 
        
        # Don't use old value in backpropagation
        In_Old_Values = tf.stop_gradient(in_old_values)
        Ex_Old_Values = tf.stop_gradient(ex_old_values)
        
        # Computing internal reward, then getting internal general advantages estimator
        Intrinsic_Rewards = tf.math.square(state_target - state_pred) / (std_in_rewards + 1e-8)
        intrinsic_advantages = self.utils.generalized_advantage_estimation(in_values, Intrinsic_Rewards, next_in_values, dones)
        Intrinsic_Returns = tf.stop_gradient(self.utils.temporal_difference(Intrinsic_Rewards, next_in_values, dones))
        
        # Getting external general advantages estimator        
        external_advantages = self.utils.generalized_advantage_estimation(ex_values, rewards, next_ex_values, dones)
        External_Returns = tf.stop_gradient(self.utils.temporal_difference(rewards, next_ex_values, dones))
        
        # Getting overall advantages
        Advantages = tf.stop_gradient((self.ex_advantages_coef * external_advantages + self.in_advantages_coef * intrinsic_advantages))
                
        # Finding the ratio (pi_theta / pi_theta__old):        
        logprobs = tf.expand_dims(self.utils.logprob(action_probs, actions), 1)         
        Old_logprobs = tf.stop_gradient(tf.expand_dims(self.utils.logprob(old_action_probs, actions), 1))
        
        # Getting entropy from the action probability 
        dist_entropy = tf.math.reduce_mean(self.utils.entropy(action_probs))
                        
        # Getting External critic loss by using Clipped critic value
        ex_vpredclipped = Ex_Old_Values + tf.clip_by_value(ex_values - Ex_Old_Values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        ex_vf_losses1 = tf.math.square(External_Returns - ex_values) # Mean Squared Error
        ex_vf_losses2 = tf.math.square(External_Returns - ex_vpredclipped) # Mean Squared Error        
        critic_ext_loss = tf.math.reduce_mean(tf.math.maximum(ex_vf_losses1, ex_vf_losses2))
                                      
        # Getting Intrinsic critic loss
        critic_int_loss = tf.math.reduce_mean(tf.math.square(Intrinsic_Returns - in_values))
        
        # Getting overall critic loss
        critic_loss = (critic_ext_loss + critic_int_loss) * 0.5
        
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
        action_probs = self.actor(state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action = self.utils.sample(action_probs)
        else:
            action = tf.math.argmax(action_probs)         
        return action    
    
    # Get loss and Do backpropagation for PPO part (the actor and critic)
    @tf.function           
    def training_ppo(self, std_in_rewards, mean_obs, std_obs, states, actions, rewards, dones, next_states):         
        with tf.GradientTape() as tape:
            loss = self.get_loss(std_in_rewards, mean_obs, std_obs, states, actions, rewards, dones, next_states)
                    
        gradients = tape.gradient(loss, self.actor.trainable_variables + self.critic_in.trainable_variables + self.critic_ex.trainable_variables)        
        self.optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.critic_in.trainable_variables + self.critic_ex.trainable_variables)) 
        
    # Get loss and Do backpropagation for RND part (the state predictor)
    @tf.function
    def training_rnd(self, obs, mean_obs, std_obs):        
        with tf.GradientTape() as tape:
            loss, intrinsic_rewards = self.get_rnd_loss(obs, mean_obs, std_obs)
                    
        gradients = tape.gradient(loss, self.rnd_predict.trainable_variables)        
        self.optimizer.apply_gradients(zip(gradients, self.rnd_predict.trainable_variables))
                                      
        return intrinsic_rewards
        
    # Update the RND part (the state predictor)
    def update_rnd(self):
        batch_size = int(self.memory.length_obs() / self.minibatch)                                      
                                      
        # Convert list in tensor
        mean_obs = tf.stop_gradient(self.memory.mean_obs)
        std_obs = tf.stop_gradient(self.memory.std_obs)
        
        # Optimize predictor for K epochs:
        for epoch in range(self.RND_epochs):  
            for obs in self.memory.get_all_obs().batch(batch_size):
                intrinsic_rewards = self.training_rnd(obs, mean_obs, std_obs) 
                    
        # Update Intrinsic Rewards Normalization Parameter
        self.updateRwdNormalizationParam(intrinsic_rewards)
        
        # Update Observation Normalization Parameter
        self.updateObsNormalizationParam()
        
        # Clear the observation
        self.memory.clearObs()
        
    # Update the PPO part (the actor and critic)
    def update_ppo(self):        
        batch_size = int(self.memory.length() / self.minibatch)
                                      
        # Getting value
        mean_obs = tf.stop_gradient(self.memory.mean_obs)
        std_obs = tf.stop_gradient(self.memory.std_obs)
        std_in_rewards = tf.stop_gradient(self.memory.std_in_rewards)
        
        # Optimize policy for K epochs:
        for epoch in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in self.memory.get_all_items().batch(batch_size):
                self.training_ppo(std_in_rewards, mean_obs, std_obs, states, actions, rewards, dones, next_states)
                    
        # Clear the memory
        self.memory.clearMemory()
                
        # Copy new weights into old policy:
        self.actor_old.set_weights(self.actor.get_weights())
        self.critic_in_old.set_weights(self.critic_in.get_weights())
        self.critic_ex_old.set_weights(self.critic_ex.get_weights())
        
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
        state_val = to_categorical(state, num_classes = state_dim) # One hot encoding for state because it's more efficient for Neural Network
        action = int(agent.act(state_val))
        state_n, reward, done, _ = env.step(action)
        
        t += 1
        t_rnd += 1        
        total_reward += reward
          
        if training_mode:
            next_state_val = to_categorical(state_n, num_classes = state_dim)  # One hot encoding for next state   
            agent.save_eps(state_val, reward, action, done, next_state_val)
            agent.save_observation(next_state_val)
            
        state = state_n     
        
        # Update the RND for every n_rnd_update
        # RND will be updated non-episodically
        if training_mode:
            if t_rnd == n_rnd_update:
                agent.update_rnd()
                #print('RND has been updated')
                t_rnd = 0
        
        if render:
            env.render()
        if done:
            return total_reward, t, t_rnd

# We run some initialization before run actual episode
# in order to normalize Observation Parameters
def run_inits_episode(env, agent, state_dim, render, n_init_episode):
    utils = Utils()
    ############################################
    env.reset()

    for i in range(n_init_episode):
        action = env.action_space.sample()
        state_n, _, done, _ = env.step(action)
        next_state_val = to_categorical(state_n, num_classes = state_dim)
        agent.save_observation(next_state_val)

        if render:
            env.render()

        if done:
            env.reset()

    agent.updateObsNormalizationParam()
    agent.memory.clearObs()
    
def main():
    try:
        register(
            id='FrozenLakeNotSlippery-v0',
            entry_point='gym.envs.toy_text:FrozenLakeEnv',
            kwargs={'map_name' : '8x8', 'is_slippery': False},
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
    reward_threshold = 1 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    
    render = False # If you want to display the image. Turn this off if you run this in Google Collab
    n_update = 1 # How many episode before you update the Policy
    n_plot_batch = 100 # How many episode you want to plot the result
    n_rnd_update = 10 # How many episode before you update the RND
    n_episode = 10000 # How many episode you want to run
    n_init_episode = 1024

    mean_obs = None
    std_obs = None
    #############################################  
    env_name = "FrozenLakeNotSlippery-v0"
    env = gym.make(env_name)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
        
    utils = Utils()     
    agent = Agent(state_dim, action_dim, training_mode)  
    ############################################# 
    
    if using_google_drive:
        from google.colab import drive
        drive.mount('/test')

    #############################################

    if training_mode:
        run_inits_episode(env, agent, state_dim, render, n_init_episode)

    #############################################
    
    rewards = []   
    batch_rewards = []
    batch_solved_reward = []
    
    times = []
    batch_times = []
    
    t_rnd = 0

    #############################################
    
    for i_episode in range(1, n_episode + 1):
        total_reward, time, t_rnd = run_episode(env, agent, state_dim, render, t_rnd, training_mode, n_rnd_update)
        print('Episode {} \t t_reward: {} \t time: {} '.format(i_episode, total_reward, time))
        batch_rewards.append(total_reward)
        batch_times.append(time) 
        
        # Update the PPO at the end of episode
        # PPO will be updated episodically
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