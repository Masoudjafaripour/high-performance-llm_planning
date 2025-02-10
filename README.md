# 🚀 High-Performance Planning with JAX, CUDA & Differentiable Simulation

## 📌 Overview
This project focuses on **high-speed path planning** using:
- **JAX for acceleration** (JIT-compilation & vectorization)
- **CUDA for parallel A*** on GPU
- **Differentiable Simulation** for gradient-based refinement
- **LLMs for heuristic enhancement** (GPT-4 guided heuristics)
- **RL for adaptive planning** (PPO-trained policies)

Additionally, we analyze **scaling laws** in:
- **Compute Efficiency** (JAX/CUDA speedup)
- **Model Scaling** (GPT-3.5 vs. GPT-4 heuristics)
- **Data Efficiency** (RL performance with increasing samples)

---

## ✨ Features
✅ **Baseline A\*** (NumPy-based)  
✅ **JAX-Accelerated A\*** (JIT-compiled for speedup)  
✅ **CUDA-Parallelized A\*** (GPU-based acceleration)  
✅ **Differentiable Path Optimization** (Gradient-based refinement)  
✅ **LLM-Guided Heuristics** (GPT-4 for heuristic improvement)  
✅ **RL-Based Adaptive Planning** (PPO-learned heuristics)  
✅ **Scaling Analysis** (Compute, model size, and data efficiency)  

---

## 📁 Project Structure
```bash
📂 high-performance-planning/
├── 📜 README.md          # Project Documentation
├── 📜 requirements.txt   # Dependencies
├── 📝 experiments/       # Benchmarking results & scaling analysis
├── 🏗️ src/               # Implementation files
│   ├── a_star.py        # Baseline A* planner
│   ├── jax_a_star.py    # JAX-accelerated A* planner
│   ├── cuda_a_star.cu   # CUDA-optimized A* planner
│   ├── differentiable_planner.py  # Differentiable path optimization
│   ├── llm_heuristics.py # LLM-based heuristic integration
│   ├── rl_planner.py    # RL-based heuristic learning
│   ├── utils.py         # Helper functions
│   └── visualize.py     # Visualization tools
└── 📜 benchmarks/       # Performance comparison results
