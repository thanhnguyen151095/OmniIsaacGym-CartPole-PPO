# PPO Algorithm for CartPole in NVIDIA Omniverse Isaac Gym Environment
This repository demonstrates how to install, build a basic task, and then build an RL agent (here is PPO) to train CartPole environments in parallel and concurrently based on Omniverse IsaacGym.

# "Why Omniverse IsaacGym?"
Omniverse Isaac Gym provides an advanced learning platform designed for efficiently training policies across diverse robotics tasks directly on GPU. The platform ensures high performance by executing both physics simulation and neural network policy training on the GPU. This is achieved through the direct transfer of data from physics buffers to PyTorch tensors, bypassing any CPU bottlenecks. As a result, the training times for complex robotics tasks are remarkably fast on a single GPU, with improvements ranging from 2 to 3 orders of magnitude when compared to conventional RL training setups that utilize a CPU-based simulator and GPU for neural networks.{https://arxiv.org/abs/2108.10470}

# How to install Omniverse Isaac Gym?
Please follow the instructions in the following link to install it. {https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html#workstation-setup}

# How to create a new RL Example (CartPole) in OmniIsaacGymEnvs
Please follow the instructions in the following link: https://docs.omniverse.nvidia.com/isaacsim/latest/isaac_gym_tutorials/tutorial_gym_isaac_gym_new_oige_example.html

# Build a Deep RL Agent (Proximal Policy Optimization (PPO))
* PPO_Buffer
  The buffer includes the following tensors:
  - states: (buffer_size, num_envs, obs_dim)
  - actions: (buffer_size, num_envs, act_dim)
  - rewards: (buffer_size, num_envs, 1)
  - terminated: (buffer_size, num_envs, 1)
  - log_prob: (buffer_size, num_envs, 1)
  - values: (buffer_size, num_envs, 1)
  - returns: (buffer_size, num_envs, 1)
  - advantages: (buffer_size, num_envs, 1)
* MPL NN
  

* PPO Algorithm


