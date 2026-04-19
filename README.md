# Meta-Reinforcement Learning for Cloud Task Scheduling (Major Project)

------------------------------------------------------------------------

# 1. Introduction

Modern cloud systems must efficiently allocate tasks across distributed
servers under dynamic workloads. Traditional heuristic schedulers like
First-Fit are fast and effective but lack adaptability. Reinforcement
Learning (RL) introduces learning-based decision-making, while Meta-RL
aims to improve adaptability across varying environments.

This project explores:

-   Heuristic scheduling baselines
-   Reinforcement Learning (Actor-Critic)
-   Meta-Reinforcement Learning (Reptile-style)
-   Comparative evaluation under stochastic workloads

------------------------------------------------------------------------

# 2. Problem Statement

Given: - N servers with CPU and RAM constraints - Incoming tasks with
varying resource requirements and durations

Objective: - Maximize utilization - Minimize queue backlog - Maximize
completed tasks - Ensure fairness (avoid starvation)

------------------------------------------------------------------------

# 3. System Architecture

## 3.1 Environment Design

-   Multi-server system
-   FIFO task queue
-   Poisson-based task arrivals
-   Continuous simulation (time steps)

### Components:

-   Task Generator
-   Server Pool
-   Scheduler (Agent)
-   Reward System

------------------------------------------------------------------------

## 3.2 State Representation

State vector includes:

-   Server CPU utilization (N values)
-   Server RAM utilization (N values)
-   Current task features:
    -   CPU requirement
    -   RAM requirement
    -   Duration
    -   Waiting time
-   Queue pressure

State dimension:

    2N + 5

------------------------------------------------------------------------

## 3.3 Action Space

Discrete:

    0 → Delay
    1..N → Assign to server i

------------------------------------------------------------------------

# 4. Reward Function

Final reward formulation:

    Reward =
        0.6 * utilization
      + 0.25 * completed_tasks
      + 0.1 * scheduled
      - 0.05 * waiting_penalty
      - 0.02 * fairness_penalty
      - 0.05 * queue_penalty

Key idea: - Encourage throughput and utilization - Penalize congestion
and delay

------------------------------------------------------------------------

# 5. Baseline Algorithms

## 5.1 Random Scheduler

-   Random action selection
-   Used as lower bound

## 5.2 First-Fit

-   Assign task to first feasible server
-   Strong heuristic baseline

## 5.3 Shortest Job First (SJF)

-   Prioritizes small tasks
-   Can cause starvation

------------------------------------------------------------------------

# 6. Reinforcement Learning

## 6.1 Model: Actor-Critic

### Architecture:

-   Shared encoder (MLP)
-   Actor head → action probabilities
-   Critic head → value estimation

## 6.2 Training Details

-   Policy Gradient with Advantage
-   Entropy Regularization
-   Gradient Clipping
-   Return Normalization

------------------------------------------------------------------------

# 7. Meta-Reinforcement Learning

## 7.1 Motivation

RL struggles to generalize across different workload distributions.

Meta-RL aims to: - Learn initialization - Adapt quickly to new
environments

------------------------------------------------------------------------

## 7.2 Task Distribution

Each task environment varies in:

-   Arrival rate (λ)
-   CPU/RAM distributions
-   Duration ranges

------------------------------------------------------------------------

## 7.3 Algorithm (Reptile-style)

### Inner Loop:

-   Train agent on a specific task

### Outer Loop:

-   Update meta-weights:

```{=html}
<!-- -->
```
    θ ← θ + α (θ_task − θ)

------------------------------------------------------------------------

# 8. Experimental Results

## 8.1 Baseline Comparison

  Model      Reward   Util   Queue    Completed
  ---------- -------- ------ -------- -----------
  Random     \~0.10   0.42   High     Low
  RL         \~0.15   0.46   Medium   Medium
  FirstFit   \~0.39   0.61   Low      High

------------------------------------------------------------------------

## 8.2 RL Observations

-   Improves over random
-   Fails to outperform First-Fit consistently
-   Sensitive to reward design

------------------------------------------------------------------------

## 8.3 Meta-RL Observations

-   Naive averaging fails
-   Reptile improves stability slightly
-   Requires proper evaluation (few-shot regime)

------------------------------------------------------------------------

# 9. Key Insights

1.  Reward shaping is critical
2.  Heuristics are strong baselines
3.  RL suffers from:
    -   High variance
    -   Limited planning ability
4.  Meta-RL requires:
    -   Structured tasks
    -   Strong inner-loop training
    -   Correct evaluation setup

------------------------------------------------------------------------

# 10. Limitations

-   No long-term planning (no sequence modeling)
-   No graph-based representation of servers
-   Meta-RL not fully optimized (MAML not implemented)
-   Evaluation sensitive to configuration

------------------------------------------------------------------------

# 11. Future Work

-   Graph Neural Networks (GNN)
-   Transformer-based scheduling
-   MAML / higher-order meta-learning
-   Real-world workload traces
-   Multi-objective optimization (latency + cost)

------------------------------------------------------------------------

# 12. How to Run

## Train RL

    python train_ac.py

## Evaluate RL

    python evaluate_rl.py

## Train Meta-RL

    python meta/meta_train.py

## Evaluate Meta-RL

    python meta/evaluate_meta.py

------------------------------------------------------------------------

# 13. Tech Stack

-   Python
-   PyTorch
-   NumPy

------------------------------------------------------------------------

# 14. Conclusion

This project demonstrates:

-   Strong baseline comparison
-   Practical RL limitations
-   Challenges in Meta-RL implementation

Key takeaway:

> Learning-based schedulers require careful design and evaluation.
> Meta-RL shows promise but is non-trivial to implement effectively.

------------------------------------------------------------------------

