# PPO Algorithm for CartPole in NVIDIA Omniverse Isaac Gym Environment
This repository demonstrates how to install, build a basic task, and then build an RL agent (here is PPO) to train CartPole environments in parallel and concurrently based on Omniverse IsaacGym.

# "Why Omniverse IsaacGym?"
Omniverse Isaac Gym provides an advanced learning platform designed for efficiently training policies across diverse robotics tasks directly on GPU. The platform ensures high performance by executing both physics simulation and neural network policy training on the GPU. This is achieved through the direct transfer of data from physics buffers to PyTorch tensors, bypassing any CPU bottlenecks. As a result, the training times for complex robotics tasks are remarkably fast on a single GPU, with improvements ranging from 2 to 3 orders of magnitude when compared to conventional RL training setups that utilize a CPU-based simulator and GPU for neural networks.
