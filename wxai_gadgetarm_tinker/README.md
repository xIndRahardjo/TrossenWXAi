# GadgetArm Experiment Replication in MuJoCo (WXAI Arm & `tinker.obj`)

This directory contains the replication of the 4-DOF Reinforcement Learning automatic grasp generation and object manipulation framework described in the research paper:

> **GadgetArm—Automatic Grasp Generation and Manipulation of 4-DOF Robot Arm for Arbitrary Objects Through Reinforcement Learning**  
> *JoungMin Park, SangYoon Lee, JaeWoon Lee, and Jumyung Um*  
> Sensors 2020, 20(21), 6183; [sensors-20-06183-v2.pdf](file:///home/kurotama/Documents/TEEP/RL_WXAI/trossen_arm_mujoco/Refference/sensors-20-06183-v2.pdf)

---

## 1. System & Experiment Architecture

- **Robot Arm**: Single **WidowX 250 AI (WXAI)** 6-DOF manipulator arm with 2 sliding gripper carriage fingers.
- **Target Object**: Custom CAD mesh **`tinker.obj`** loaded dynamically with scale `0.003` (~5-6 cm bounding extent), initialized on a tabletop surface with collision physics and freejoint 6-DOF dynamics.
- **Simulation Environment**: MuJoCo 3.8.0.

### Decoupled 3-Stage Reward Function
1. **Approach Stage**:
   $$R_{\text{approach}} = -10.0 \cdot d(\text{ee}, \text{tinker}) + 50.0 \cdot (d_{\text{prev}} - d_{\text{curr}})$$
2. **Gripping Stage (Contact Verification)**:
   $$R_{\text{grasp}} = +100.0 \quad \text{when both WXAI gripper fingers make contact with tinker.obj}$$
3. **Lift & Gravity Transport Stage**:
   $$R_{\text{lift}} = +200.0 \cdot (z_{\text{obj}} - z_{\text{table}}) + 50.0 \cdot \Delta d(\text{obj}, \text{target}) \quad \text{for } z_{\text{obj}} > z_{\text{table}} + 0.03$$
   - Early episode termination and drop penalty ($R_{\text{fail}} = -100.0$) if the object falls or slips below table height.
4. **Smoothness Penalty**:
   $$R_{\text{smooth}} = -0.005 \cdot \|\dot{q}\|^2 - 0.0005 \cdot \|\ddot{q}\|^2$$

---

## 2. Hyperparameter Configuration (Matched to Paper Table 1)

| Parameter | Value | Description |
|---|---|---|
| **RL Algorithm** | PPO (Proximal Policy Optimization) | Continuous Action Control |
| **Learning Rate** | $3.0 \times 10^{-4}$ | Adam Optimizer |
| **Batch Size** | 800 | Mini-batch optimization size |
| **Buffer Size** | 12,000 | Timesteps collected per PPO iteration |
| **Max Steps** | 500,000 | Total training timesteps |
| **Beta (Entropy Coeff)** | $5.0 \times 10^{-3}$ | Policy entropy regularization |
| **Epsilon (Clip Ratio)** | 0.2 | PPO surrogate clipping threshold |
| **Lambda (GAE)** | 0.95 | Generalized Advantage Estimation |
| **Gamma** | 0.99 | Discount factor |

---

## 3. Directory & File Overview

- `scene_tinker_gadgetarm.xml`: MuJoCo XML scene loading `wxai_base.xml`, tabletop surface, target site, and `tinker.obj` mesh asset.
- `wxai_gadgetarm_env.py`: Gymnasium-compatible MuJoCo environment class (`WxaiGadgetArmEnv`).
- `ppo_agent.py`: PyTorch PPO Agent implementation with clipped surrogate loss and GAE.
- `train_gadgetarm.py`: 500,000 timestep training script with model checkpointing and JSON logging.
- `evaluate_gadgetarm.py`: Evaluation pipeline producing publication-quality plots in `plots/`.
- `plots/`: Output plots for rewards, grasp success, and trajectory elevation profiles.

---

## 4. How to Run

### Run Training (500,000 Steps)
```bash
/home/kurotama/miniconda3/envs/trossen_mujoco_env/bin/python train_gadgetarm.py --total_steps 500000
```

### Run Quick Training Test (e.g. 24,000 Steps)
```bash
/home/kurotama/miniconda3/envs/trossen_mujoco_env/bin/python train_gadgetarm.py --total_steps 24000
```

### Run Evaluation & Generate Plots
```bash
/home/kurotama/miniconda3/envs/trossen_mujoco_env/bin/python evaluate_gadgetarm.py --episodes 20 --video
```
