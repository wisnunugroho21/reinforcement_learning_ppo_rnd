

# PPO-RND

Simple code to demonstrate Reinforcement Learning using Proximal Policy Optimization and Random Network Distillation

## Getting Started

This project is using Pytorch for Deep Learning Framework and Gym for Reinforcement Learning Environment.  
Although it's not required, but i recommend run this project on a PC with GPU and 8 GB Ram

### Prerequisites

Make sure you have installed Pytorch and Gym.  
- Click [here](https://pytorch.org/get-started/locally/) to install pytorch  
- Click [here](https://gym.openai.com/docs/) to install gym

### Installing

Just clone this project into your work folder

```
git clone https://github.com/wisnunugroho21/reinforcement_learning_ppo_rnd.git
```

## Running the project

After you clone the project, run following script in cmd/terminal :

```
cd reinforcement_learning_ppo_rnd/PPO_RND
python ppo_rnd_cartpole_pytorch.py
```

## Proximal Policy Optimization

PPO is motivated by the same question as TRPO: how can we take the biggest possible improvement step on a policy using the data we currently have, without stepping so far that we accidentally cause performance collapse? Where TRPO tries to solve this problem with a complex second-order method, PPO is a family of first-order methods that use a few other tricks to keep new policies close to old. PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

There are two primary variants of PPO: PPO-Penalty and PPO-Clip.

* PPO-Penalty approximately solves a KL-constrained update like TRPO, but penalizes the KL-divergence in the objective function instead of making it a hard constraint, and automatically adjusts the penalty coefficient over the course of training so that it’s scaled appropriately.

* PPO-Clip doesn’t have a KL-divergence term in the objective and doesn’t have a constraint at all. Instead relies on specialized clipping in the objective function to remove incentives for the new policy to get far from the old policy.

This repository use PPO-Clip  
You can read full detail of PPO in [here](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

## Random Network Distillation

Random Network Distillation (RND), a prediction-based method for encouraging reinforcement learning agents to explore their environments through curiosity, which for the first time exceeds average human performance on Montezuma’s Revenge. RND achieves state-of-the-art performance, periodically finds all 24 rooms and solves the first level without using demonstrations or having access to the underlying state of the game.  

RND incentivizes visiting unfamiliar states by measuring how hard it is to predict the output of a fixed random neural network on visited states. In unfamiliar states it’s hard to guess the output, and hence the reward is high. It can be applied to any reinforcement learning algorithm, is simple to implement and efficient to scale. 

You can read full detail of RND in [here](https://openai.com/blog/reinforcement-learning-with-prediction-based-rewards/)

## Result

### LunarLander using PPO (Non RND)

| Result Gif  | Award Progress Graph |
| ------------- | ------------- |
| ![Result Gif](https://github.com/wisnunugroho21/reinforcement_learning_ppo_rnd/blob/master/Result/giphy1.gif)  | ![Award Progress Graph](https://github.com/wisnunugroho21/reinforcement_learning_ppo_rnd/blob/master/Result/lunarlander_ppo.png)  |

## Future Development

I really want to adapt this code to Tensorflow  
But I think it's better to wait for Tensorflow 2.0 to be fully released

For now, I focus on how to implement this project on more difficult environment (Atari games, Roboschool, etc)

## Contributing

This project is far from finish and will be improved anytime . Any fix, contribute, or idea would be very appreciated
