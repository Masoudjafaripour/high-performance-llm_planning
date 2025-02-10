# 🚀 High-Performance Planning with JAX, CUDA & Differentiable Simulation

## 📌 Overview
This project explores **high-performance path planning** using **JAX, CUDA, and Differentiable Simulation**, integrating **LLMs for heuristic improvement** and **RL for adaptive planning**. We analyze **scaling laws in compute efficiency, model size, and data efficiency** to understand how performance improves with increasing computational resources.

## ✨ Features
✅ **Baseline A\* Path Planning** (NumPy-based)  
✅ **JAX-Accelerated A\*** (JIT-compiled for speedup)  
✅ **CUDA-Parallelized A\*** (Fast path planning on GPU)  
✅ **Differentiable Path Optimization** (Gradient-based refinement)  
✅ **LLM-Guided Heuristic Learning** (GPT-4 suggests heuristics)  
✅ **RL-Based Adaptive Heuristics** (Train PPO to learn better planning strategies)  
✅ **Scaling Analysis** (Compute, model size, and data scaling laws)  

## 📁 Project Structure
📂 high-performance-planning/ ├── 📜 README.md # Project Documentation ├── 📜 requirements.txt # Dependencies ├── 📝 experiments/ # Benchmarking results & scaling analysis ├── 🏗️ src/ # Implementation files │ ├── a_star.py # Baseline A* planner │ ├── jax_a_star.py # JAX-accelerated A* planner │ ├── cuda_a_star.cu # CUDA-optimized A* planner │ ├── differentiable_planner.py # Differentiable path optimization │ ├── llm_heuristics.py # LLM-based heuristic integration │ ├── rl_planner.py # RL-based heuristic learning │ ├── utils.py # Helper functions │ └── visualize.py # Visualization tools └── 📜 benchmarks/ # Performance comparison results
