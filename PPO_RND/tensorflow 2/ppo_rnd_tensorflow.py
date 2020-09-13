import gym
from gym.envs.registration import register
    
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical

import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy

class Utils():
    def prepro(self, I):
        I           = I[35:195] # crop
        I           = I[::2,::2, 0] # downsample by factor of 2
        I[I == 144] = 0 # erase background (background type 1)
        I[I == 109] = 0 # erase background (background type 2)
        I[I != 0]   = 1 # everything else (paddles, ball) just set to 1
        X           = I.astype(np.float32).ravel() # Combine items in 1 array 
        return X

    def countNewMean(self, prevMean, prevLen, newData):
        return ((prevMean * prevLen) + tf.math.reduce_sum(newData, 0)) / (prevLen + newData.shape[0])
      
    def countNewStd(self, prevStd, prevLen, newData):
        return tf.math.sqrt(((tf.math.square(prevStd) * prevLen) + (tf.math.reduce_variance(newData, 0) * newData.shape[0])) / (prevStd + newData.shape[0]))

    def normalize(self, data, mean = None, std = None, clip = None):
        if isinstance(mean, tf.Tensor) and isinstance(std, tf.Tensor):
            data_normalized = (data - mean) / (std + 1e-8)            
        else:
            data_normalized = (data - tf.math.reduce_mean(data)) / (tf.math.reduce_std(data) + 1e-8)
                    
        if clip :
            data_normalized = tf.clip_by_value(data_normalized, -clip, clip)

        return data_normalized

class Actor_Model(Model):
  def __init__(self, state_dim, action_dim):
    super(Actor_Model, self).__init__()
    self.d1     = Dense(64, activation='relu')
    self.d2     = Dense(64, activation='relu')
    self.dout   = Dense(action_dim, activation='softmax')

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.dout(x)

class Critic_Model(Model):
  def __init__(self, state_dim, action_dim):
    super(Critic_Model, self).__init__()
    self.d1     = Dense(64, activation='relu')
    self.d2     = Dense(64, activation='relu')
    self.dout   = Dense(1, activation='linear')

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.dout(x)

class RND_Model(Model):
  def __init__(self, state_dim, action_dim):
    super(RND_Model, self).__init__()
    self.d1     = Dense(64, activation='relu')
    self.d2     = Dense(64, activation='relu')
    self.dout   = Dense(5, activation='linear')

  def call(self, x):
    x = self.d1(x)
    x = self.d2(x)
    return self.dout(x)

class ObsMemory():
    def __init__(self, state_dim):
        self.observations    = []

        self.mean_obs           = tf.zeros(state_dim, dtype = tf.float32)
        self.std_obs            = tf.zeros(state_dim, dtype = tf.float32)
        self.std_in_rewards     = tf.zeros(1, dtype = tf.float32)
        self.total_number_obs   = tf.zeros(1, dtype = tf.float32)
        self.total_number_rwd   = tf.zeros(1, dtype = tf.float32)

    def __len__(self):
        return len(self.observations)

    def get_all_items(self):
        observations = tf.constant(self.observations, dtype = tf.float32)        
        return tf.data.Dataset.from_tensor_slices(observations)

    def save_eps(self, obs):
        self.observations.append(obs)

    def save_observation_normalize_parameter(self, mean_obs, std_obs, total_number_obs):
        self.mean_obs           = mean_obs
        self.std_obs            = std_obs
        self.total_number_obs   = total_number_obs
        
    def save_rewards_normalize_parameter(self, std_in_rewards, total_number_rwd):
        self.std_in_rewards     = std_in_rewards
        self.total_number_rwd   = total_number_rwd

    def clearMemory(self):
        del self.observations[:]

class Memory():
    def __init__(self):
        self.actions        = [] 
        self.states         = []
        self.rewards        = []
        self.dones          = []     
        self.next_states    = []

    def __len__(self):
        return len(self.dones)

    def get_all_items(self):
        states = tf.constant(self.states, dtype = tf.float32)
        actions = tf.constant(self.actions, dtype = tf.float32)
        rewards = tf.expand_dims(tf.constant(self.rewards, dtype = tf.float32), 1)
        dones = tf.expand_dims(tf.constant(self.dones, dtype = tf.float32), 1)
        next_states = tf.constant(self.next_states, dtype = tf.float32)
        
        return tf.data.Dataset.from_tensor_slices((states, actions, rewards, dones, next_states))      

    def save_eps(self, state, action, reward, done, next_state):
        self.rewards.append(reward)
        self.states.append(state)
        self.actions.append(action)
        self.dones.append(done)
        self.next_states.append(next_state)        

    def clearMemory(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.dones[:]
        del self.next_states[:]  

class Distributions():
    def sample(self, datas):
        distribution = tfp.distributions.Categorical(probs = datas)
        return distribution.sample()
        
    def entropy(self, datas):
        distribution = tfp.distributions.Categorical(probs = datas)            
        return distribution.entropy()
      
    def logprob(self, datas, value_data):
        distribution = tfp.distributions.Categorical(probs = datas)
        return tf.expand_dims(distribution.log_prob(value_data), 1)

    def kl_divergence(self, datas1, datas2):
        distribution1 = tfp.distributions.Categorical(probs = datas1)
        distribution2 = tfp.distributions.Categorical(probs = datas2)

        return tf.expand_dims(tfp.distributions.kl_divergence(distribution1, distribution2), 1)

class PolicyFunction():
    def __init__(self, gamma = 0.99, lam = 0.95):
        self.gamma  = gamma
        self.lam    = lam

    def monte_carlo_discounted(self, datas):
        returns     = []        
        running_add = 0
        
        for i in reversed(range(len(datas))):
            running_add = running_add * self.gamma + datas[i]
            returns.insert(0, running_add)
            
        return tf.stack(returns)
      
    def temporal_difference(self, reward, next_value, done):
        q_values = reward + (1 - done) * self.gamma * next_value           
        return q_values
      
    def generalized_advantage_estimation(self, values, rewards, next_values, dones):
        gae     = 0
        adv     = []     

        delta   = rewards + (1.0 - dones) * self.gamma * next_values - values          
        for step in reversed(range(len(rewards))):  
            gae = delta[step] + (1.0 - dones[step]) * self.gamma * self.lam * gae
            adv.insert(0, gae)
            
        return tf.stack(adv)

class Agent():  
    def __init__(self, state_dim, action_dim, is_training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                 minibatch, PPO_epochs, gamma, lam, learning_rate):        
        self.policy_kl_range        = policy_kl_range 
        self.policy_params          = policy_params
        self.value_clip             = value_clip    
        self.entropy_coef           = entropy_coef
        self.vf_loss_coef           = vf_loss_coef
        self.minibatch              = minibatch       
        self.PPO_epochs             = PPO_epochs
        self.RND_epochs             = 5
        self.is_training_mode       = is_training_mode
        self.action_dim             = action_dim               

        self.actor                  = Actor_Model(state_dim, action_dim)
        self.actor_old              = Actor_Model(state_dim, action_dim)

        self.ex_critic              = Critic_Model(state_dim, action_dim)
        self.ex_critic_old          = Critic_Model(state_dim, action_dim)

        self.in_critic              = Critic_Model(state_dim, action_dim)
        self.in_critic_old          = Critic_Model(state_dim, action_dim)

        self.rnd_predict            = RND_Model(state_dim, action_dim)
        self.rnd_target             = RND_Model(state_dim, action_dim)

        self.ppo_optimizer          = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        self.rnd_optimizer          = tf.keras.optimizers.Adam(learning_rate = learning_rate)

        self.memory                 = Memory()
        self.obs_memory             = ObsMemory(state_dim)
        self.utils                  = Utils()

        self.policy_function        = PolicyFunction(gamma, lam)  
        self.distributions          = Distributions()

        self.ex_advantages_coef     = 2
        self.in_advantages_coef     = 1        
        self.clip_normalization     = 5

    def save_eps(self, state, action, reward, done, next_state):
        self.memory.save_eps(state, action, reward, done, next_state)

    def save_observation(self, obs):
        self.obs_memory.save_eps(obs)

    def updateObsNormalizationParam(self, obs):
        obs                 = tf.constant(obs, dtype = tf.float32)

        mean_obs            = self.utils.countNewMean(self.obs_memory.mean_obs, self.obs_memory.total_number_obs, obs)
        std_obs             = self.utils.countNewStd(self.obs_memory.std_obs, self.obs_memory.total_number_obs, obs)
        total_number_obs    = len(obs) + self.obs_memory.total_number_obs
        
        self.obs_memory.save_observation_normalize_parameter(mean_obs, std_obs, total_number_obs)
    
    def updateRwdNormalizationParam(self, in_rewards):
        std_in_rewards      = self.utils.countNewStd(self.obs_memory.std_in_rewards, self.obs_memory.total_number_rwd, in_rewards)
        total_number_rwd    = len(in_rewards) + self.obs_memory.total_number_rwd
        
        self.obs_memory.save_rewards_normalize_parameter(std_in_rewards, total_number_rwd)

    # Loss for RND 
    def get_rnd_loss(self, state_pred, state_target):        
        # Don't update target state value
        state_target = tf.stop_gradient(state_target)        
        
        # Mean Squared Error Calculation between state and predict
        forward_loss = tf.math.reduce_mean(tf.math.square(state_target - state_pred) * 0.5)
        return forward_loss

    # Loss for PPO  
    def get_PPO_loss(self, action_probs, ex_values, old_action_probs, old_ex_values, next_ex_values, actions, ex_rewards, dones, 
        state_preds, state_targets, in_values, old_in_values, next_in_values, std_in_rewards):   

        # Don't use old value in backpropagation
        Old_ex_values           = tf.stop_gradient(old_ex_values)

        # Getting general advantages estimator
        External_Advantages     = self.policy_function.generalized_advantage_estimation(ex_values, ex_rewards, next_ex_values, dones)
        External_Returns        = tf.stop_gradient(External_Advantages + ex_values)
        External_Advantages     = tf.stop_gradient((External_Advantages - tf.math.reduce_mean(External_Advantages)) / (tf.math.reduce_std(External_Advantages) + 1e-6))

        # Computing internal reward, then getting internal general advantages estimator
        in_rewards              = tf.math.square(state_targets - state_preds) * 0.5 / (tf.math.reduce_mean(std_in_rewards) + 1e-8)
        Internal_Advantages     = self.policy_function.generalized_advantage_estimation(in_values, in_rewards, next_in_values, dones)
        Internal_Returns        = tf.stop_gradient(Internal_Advantages + in_values)
        Internal_Advantages     = tf.stop_gradient((Internal_Advantages - tf.math.reduce_mean(Internal_Advantages)) / (tf.math.reduce_std(Internal_Advantages) + 1e-6))

        # Getting overall advantages
        Advantages              = tf.stop_gradient(self.ex_advantages_coef * External_Advantages + self.in_advantages_coef * Internal_Advantages)

        # Finding the ratio (pi_theta / pi_theta__old):        
        logprobs        = self.distributions.logprob(action_probs, actions)
        Old_logprobs    = tf.stop_gradient(self.distributions.logprob(old_action_probs, actions))
        ratios          = tf.math.exp(logprobs - Old_logprobs) # ratios = old_logprobs / logprobs

        # Finding KL Divergence                
        Kl              = self.distributions.kl_divergence(old_action_probs, action_probs)

        # Combining TR-PPO with Rollback (Truly PPO)
        pg_loss         = tf.where(
                tf.logical_and(Kl >= self.policy_kl_range, ratios >= 1),
                ratios * Advantages - self.policy_params * Kl,
                ratios * Advantages - self.policy_kl_range
        )
        pg_loss         = tf.math.reduce_mean(pg_loss)

        # Getting entropy from the action probability
        dist_entropy    = tf.math.reduce_mean(self.distributions.entropy(action_probs))

        # Getting critic loss by using Clipped critic value
        ex_vpredclipped = Old_ex_values + tf.clip_by_value(ex_values - Old_ex_values, -self.value_clip, self.value_clip) # Minimize the difference between old value and new value
        ex_vf_losses1   = tf.math.square(External_Returns - ex_values) # Mean Squared Error
        ex_vf_losses2   = tf.math.square(External_Returns - ex_vpredclipped) # Mean Squared Error
        critic_ext_loss = tf.math.reduce_mean(tf.math.maximum(ex_vf_losses1, ex_vf_losses2))  

        # Getting Intrinsic critic loss
        critic_int_loss = tf.math.reduce_mean(tf.math.square(Internal_Returns - in_values))

        # Getting overall critic loss
        critic_loss     = (critic_ext_loss + critic_int_loss) * 0.5

        # We need to maximaze Policy Loss to make agent always find Better Rewards
        # and minimize Critic Loss 
        loss            = (critic_loss * self.vf_loss_coef) - (dist_entropy * self.entropy_coef) - pg_loss
        return loss       

    @tf.function
    def act(self, state):
        state           = tf.expand_dims(tf.cast(state, dtype = tf.float32), 0)
        action_probs    = self.actor(state)
        
        # We don't need sample the action in Test Mode
        # only sampling the action in Training Mode in order to exploring the actions
        if self.is_training_mode:
            # Sample the action
            action  = self.distributions.sample(action_probs) 
        else:
            action  = tf.math.argmax(action_probs, 1)  
              
        return action

    # Get loss and Do backpropagation
    @tf.function
    def training_rnd(self, obs, mean_obs, std_obs):
        obs             = self.utils.normalize(obs, mean_obs, std_obs)
        
        with tf.GradientTape() as tape:
            state_pred      = self.rnd_predict(obs)
            state_target    = self.rnd_target(obs)

            loss            = self.get_rnd_loss(state_pred, state_target)

        gradients = tape.gradient(loss, self.rnd_predict.trainable_variables)        
        self.rnd_optimizer.apply_gradients(zip(gradients, self.rnd_predict.trainable_variables))

        return (state_target - state_pred)

    # Get loss and Do backpropagation
    @tf.function
    def training_ppo(self, states, actions, rewards, dones, next_states, mean_obs, std_obs, std_in_rewards):     
        # Don't update rnd value
        obs             = tf.stop_gradient(self.utils.normalize(next_states, mean_obs, std_obs, self.clip_normalization))
        state_preds     = self.rnd_predict(obs)
        state_targets   = self.rnd_target(obs)

        with tf.GradientTape() as tape:
            action_probs, ex_values, in_values                  = self.actor(states), self.ex_critic(states),  self.in_critic(states)
            old_action_probs, old_ex_values, old_in_values      = self.actor_old(states), self.ex_critic_old(states),  self.in_critic_old(states)
            next_ex_values, next_in_values                      = self.ex_critic(next_states),  self.in_critic(next_states)

            loss            = self.get_PPO_loss(action_probs, ex_values, old_action_probs, old_ex_values, next_ex_values, actions, rewards, dones,
                                state_preds, state_targets, in_values, old_in_values, next_in_values, std_in_rewards)

        gradients = tape.gradient(loss, self.actor.trainable_variables + self.ex_critic.trainable_variables + self.in_critic.trainable_variables)        
        self.ppo_optimizer.apply_gradients(zip(gradients, self.actor.trainable_variables + self.ex_critic.trainable_variables + self.in_critic.trainable_variables)) 

    # Update the model
    def update_rnd(self):        
        batch_size  = int(len(self.obs_memory) / self.minibatch)      

        # Optimize policy for K epochs:
        intrinsic_rewards = 0
        for _ in range(self.RND_epochs):       
            for obs in self.obs_memory.get_all_items().batch(batch_size):
                intrinsic_rewards = self.training_rnd(obs, self.obs_memory.mean_obs, self.obs_memory.std_obs)       

        self.updateObsNormalizationParam(self.obs_memory.observations)
        self.updateRwdNormalizationParam(intrinsic_rewards)

        # Clear the memory
        self.obs_memory.clearMemory()

    # Update the model
    def update_ppo(self):        
        batch_size = int(len(self.memory) / self.minibatch)

        # Optimize policy for K epochs:
        for _ in range(self.PPO_epochs):       
            for states, actions, rewards, dones, next_states in self.memory.get_all_items().batch(batch_size):
                self.training_ppo(states, actions, rewards, dones, next_states,
                    self.obs_memory.mean_obs, self.obs_memory.std_obs, self.obs_memory.std_in_rewards)

        # Clear the memory
        self.memory.clearMemory()

        # Copy new weights into old policy:
        self.actor_old.set_weights(self.actor.get_weights())
        self.ex_critic_old.set_weights(self.ex_critic.get_weights())
        self.in_critic_old.set_weights(self.in_critic.get_weights())

    def save_weights(self):
        self.actor.save_weights('bipedalwalker_w/actor_ppo', save_format='tf')
        self.actor_old.save_weights('bipedalwalker_w/actor_old_ppo', save_format='tf')

        self.ex_critic.save_weights('bipedalwalker_w/ex_critic_ppo', save_format='tf')
        self.ex_critic_old.save_weights('bipedalwalker_w/ex_critic_old_ppo', save_format='tf')

        self.in_critic.save_weights('bipedalwalker_w/in_critic_ppo', save_format='tf')
        self.in_critic_old.save_weights('bipedalwalker_w/in_critic_old_ppo', save_format='tf')
        
    def load_weights(self):
        self.actor.load_weights('bipedalwalker_w/actor_ppo')
        self.actor_old.load_weights('bipedalwalker_w/actor_old_ppo')

        self.ex_critic.load_weights('bipedalwalker_w/ex_critic_ppo')
        self.ex_critic_old.load_weights('bipedalwalker_w/ex_critic_old_ppo')

        self.in_critic.load_weights('bipedalwalker_w/in_critic_ppo')
        self.in_critic_old.load_weights('bipedalwalker_w/in_critic_old_ppo')

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

def run_inits_episode(env, agent, state_dim, render, n_init_episode):
    ############################################
    env.reset()

    for _ in range(n_init_episode):
        action                  = env.action_space.sample()
        next_state, _, done, _  = env.step(action)
        next_state              = to_categorical(next_state, num_classes = state_dim)
        agent.save_observation(next_state)

        if render:
            env.render()

        if done:
            env.reset()

    agent.updateObsNormalizationParam(agent.obs_memory.observations)
    agent.memory.clearObs()

    return agent

def run_episode(env, agent, state_dim, render, training_mode, t_updates, n_update):
    ############################################
    state           = to_categorical(env.reset(), num_classes = state_dim)
    done            = False
    total_reward    = 0
    eps_time        = 0
    ############################################
    
    while not done:
        action                      = int(agent.act(state))
        next_state, reward, done, _ = env.step(action)
        next_state                  = to_categorical(next_state, num_classes = state_dim)
        
        eps_time        += 1 
        t_updates       += 1
        total_reward    += reward

        if training_mode:
            agent.save_eps(state.tolist(), float(action), float(reward), float(done), next_state.tolist())
            agent.save_observation(next_state)
            
        state   = next_state
                
        if render:
            env.render()
        
        if training_mode:
            if t_updates % n_update == 0:
                agent.update_rnd()
                t_updates = 0
                #print('update rnd')
        
        if done:           
            return total_reward, eps_time, t_updates           

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
    load_weights        = False # If you want to load the agent, set this to True
    save_weights        = False # If you want to save the agent, set this to True
    training_mode       = True # If you want to train the agent, set this to True. But set this otherwise if you only want to test it
    reward_threshold    = 300 # Set threshold for reward. The learning will stop if reward has pass threshold. Set none to sei this off
    using_google_drive  = False

    render              = False # If you want to display the image, set this to True. Turn this off if you run this in Google Collab
    n_step_update       = 32 # How many episode before you update the Policy. Recommended set to 128 for Discrete
    n_eps_update        = 10
    n_plot_batch        = 100000000 # How many episode you want to plot the result
    n_episode           = 100000 # How many episode you want to run
    n_saved             = 10 # How many episode to run before saving the weights

    policy_kl_range     = 0.0008 # Recommended set to 0.0008 for Discrete
    policy_params       = 20 # Recommended set to 20 for Discrete
    value_clip          = 1.0 # How many value will be clipped. Recommended set to the highest or lowest possible reward
    entropy_coef        = 0.05 # How much randomness of action you will get
    vf_loss_coef        = 1.0 # Just set to 1
    minibatch           = 1 # How many batch per update. size of batch = n_update / minibatch. Recommended set to 4 for Discrete
    PPO_epochs          = 4 # How many epoch per update. Recommended set to 10 for Discrete
    
    gamma               = 0.99 # Just set to 0.99
    lam                 = 0.95 # Just set to 0.95
    learning_rate       = 2.5e-4 # Just set to 0.95
    ############################################# 
    env_name            = 'FrozenLakeNotSlippery-v0' # Set the env you want
    env                 = gym.make(env_name)

    state_dim           = env.observation_space.n
    action_dim          = env.action_space.n

    agent               = Agent(state_dim, action_dim, training_mode, policy_kl_range, policy_params, value_clip, entropy_coef, vf_loss_coef,
                            minibatch, PPO_epochs, gamma, lam, learning_rate)  
    #############################################     
    if using_google_drive:
        from google.colab import drive
        drive.mount('/test')

    if load_weights:
        agent.load_weights()
        print('Weight Loaded')

    rewards             = []   
    batch_rewards       = []
    batch_solved_reward = []

    times               = []
    batch_times         = []

    t_updates           = 0

    for i_episode in range(1, n_episode + 1):
        total_reward, time, t_updates = run_episode(env, agent, state_dim, render, training_mode, t_updates, n_step_update)
        print('Episode {} \t t_reward: {} \t time: {} \t '.format(i_episode, total_reward, time))
        batch_rewards.append(int(total_reward))
        batch_times.append(time)

        if i_episode % n_eps_update == 0:
            agent.update_ppo()
            #print('update ppo')           

        if save_weights:
            if i_episode % n_saved == 0:
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

            batch_rewards   = []
            batch_times     = []

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