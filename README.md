# 🎮 Deep Q-Learning Agent for Atari Video Pinball 

## 🧠 Overview
This project implements a **Deep Q-Learning (DQN)** agent using **PyTorch** to play the **Atari “Video Pinball”** environment from the **OpenAI Gymnasium ALE** suite.  
The goal is to train an autonomous agent that learns optimal control strategies purely through **trial and error**, using raw pixel input without any predefined rules.

This implementation replaces Keras-RL with a **from-scratch PyTorch version**, featuring complete control over training logic, replay memory, epsilon scheduling, and visualization.

---

## 🚀 Key Features
- **PyTorch-based Deep Q-Learning architecture**
- **Custom Environment Wrappers**:
  - Frame skipping with **max-pooling** to prevent flicker  
  - Grayscale + resizing preprocessing  
  - Frame stacking for temporal context  
- **Experience Replay Buffer** and **Target Network Updates**
- **Epsilon-Greedy Exploration** with exponential decay
- **GPU Acceleration** for efficient training
- **Live Reward Tracking & Visualization**
- **Video Rendering** of trained agent gameplay

---

---

## 📊 Approach

### 1. Environment Setup
- Environment: `ALE/VideoPinball-v5`  
- Applied custom wrappers for:
  - Frame skipping (4 steps, with max-pooling)
  - Frame preprocessing (grayscale, 84x84)
  - Frame stacking (4-frame temporal state)

### 2. Model Architecture
- **3 Convolutional Layers** → **2 Fully Connected Layers**
- Input: `4 x 84 x 84` frames  
- Output: Q-values for all possible actions  
- Activation: ReLU  
- Optimizer: Adam (`lr=1e-4`)

### 3. Learning Strategy
- **Replay Buffer Size:** 100,000 transitions  
- **Batch Size:** 32  
- **Gamma (Discount Factor):** 0.99  
- **Epsilon Decay:** From 1.0 → 0.1 over episodes  
- **Target Network Update:** Every 1000 steps  
- **Episodes:** 500 (can be tuned)

### 4. Training Loop
- For each step:
  1. Select an action (ε-greedy policy)  
  2. Execute and store transition in memory  
  3. Sample a mini-batch from replay buffer  
  4. Compute loss = `(Q_expected - Q_target)^2`  
  5. Optimize policy network and periodically sync target network

### 5. Evaluation
- After training, the agent is tested with **greedy policy** (no exploration).  
- Gameplay videos are generated using `imageio` to visually demonstrate learned strategies.

---

## 📈 Results
- The DQN agent showed **consistent improvement in total episode rewards** over time.  
- Early episodes displayed random flipper movements; later ones demonstrated ball-tracking behavior.  
- Reward curve stabilized after ~400 episodes, showing convergence toward an effective policy.  
- The trained model successfully interacts with the ball and avoids losing turns frequently.

---

## 📚 References
- **Mnih et al. (2015)** – *“Human-level control through deep reinforcement learning.”*  
- **Hasselt et al. (2016)** – *“Deep Reinforcement Learning with Double Q-learning.”*  
- **Hessel et al. (2018)** – *“Rainbow: Combining Improvements in Deep Reinforcement Learning.”*  
- PyTorch & Gymnasium Documentation  
- Additional conceptual clarifications taken with the help of **Anthropic Claude**.

---



## ⚖️ License
This repository is released under the **MIT License**.

