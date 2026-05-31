# 3SPO_GUI: State-Score-Supervised Policy Optimization for GUI Agent

<p align="center">
  <a href="#readme-cn">中文文档</a> | <a href="README_CN.md">简体中文 README</a>
</p>

**3SPO_GUI** is a reinforcement learning framework for training GUI agents in real desktop environments. Unlike trajectory-level RL methods (GRPO, PPO) that only reward the final task outcome, 3SPO performs **step-level policy optimization** — it evaluates and learns from each individual GUI action at every state, enabling finer-grained credit assignment and more sample-efficient training.

The framework is built on [veRL](https://github.com/volcengine/verl) and [OSWorld](https://github.com/xlang-ai/OSWorld), using FSDP + vLLM for distributed training and Apptainer-based Ubuntu VMs as interactive desktop environments.

## Key Features

- **3SPO Algorithm** — a novel step-level RL method with visual state identification, adaptive rollout budget, and DFS-based exploration
- **Real Desktop Environments** — agents interact with Ubuntu VMs (Chrome, GIMP, LibreOffice, VS Code, etc.) via screenshots and pyautogui actions
- **Multi-modal VLM Support** — native support for Qwen2.5-VL and UI-TARS vision-language models
- **Distributed Training** — FSDP for model parallelism + Ray for orchestration + vLLM for fast rollout inference
- **Multiple RL Baselines** — GRPO, GAE, RLOO, REINFORCE++, ReMax, and ARPO all available out of the box
- **OSWorld Benchmark** — 369-task evaluation across 10 desktop application domains

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     RayPPOTrainer (Driver)                    │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────────────┐ │
│  │ 3SPO DFS│  │ State Manager│  │ Advantage & Reward Comp. │ │
│  │ Explorer│  │ (ψ, VID, S)  │  │  (step-level 3SPO/GRPO)  │ │
│  └────┬────┘  └──────────────┘  └──────────────────────────┘ │
└───────┼──────────────────────────────────────────────────────┘
        │ Ray RPC
        ▼
┌───────────────────────────────────────────────────────────────┐
│                    Worker Groups (Ray Actors)                  │
│                                                               │
│  ┌──────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ ActorRollout     │  │ Ref Policy   │  │ Env Workers     │  │
│  │ (FSDP + vLLM)    │  │ (FSDP)       │  │ (Ubuntu VMs ×N) │  │
│  └──────────────────┘  └──────────────┘  └────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

## Supported Algorithms

| Algorithm | Granularity | Description |
|-----------|-------------|-------------|
| **3SPO** | Step-level | Visual state scoring + adaptive rollout + DFS exploration |
| GRPO | Trajectory-level | Group Relative Policy Optimization with normalized rewards |
| GAE | Step-level | Generalized Advantage Estimation (requires critic network) |
| RLOO | Trajectory-level | Leave-one-out baseline |
| REINFORCE++ | Trajectory-level | Discounted returns with masked whitening |
| ReMax | Trajectory-level | Greedy rollout as reward baseline |
| ARPO | Trajectory-level | GRPO + online replay buffer for positive sample injection |

## Installation

### Prerequisites

- Python >= 3.10
- CUDA >= 12.1
- Apptainer / Singularity (for Ubuntu VM environments)
- Ray (for distributed orchestration)

### Setup

```bash
# Clone repository with submodule
git clone --recursive https://github.com/your-org/3SPO-GUI.git
cd 3SPO-GUI

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Initialize OSWorld submodule (if not cloned with --recursive)
git submodule update --init --recursive
```

### Environment Setup

The GUI environment runs Ubuntu inside Apptainer containers. You will need:

1. **Ubuntu VM image** (`Ubuntu.qcow2`) — the disk image for the desktop environment
2. **Apptainer image / Sandbox** — the container runtime for OSWorld

```bash
# (Optional) Build sandbox from .sif file if needed
apptainer build --sandbox osworld_sandbox osworld.sif
```

Set the paths to your VM image and Apptainer sandbox in the evaluation scripts or as environment variables:

```bash
export LOCAL_IMAGE=/path/to/Ubuntu.qcow2
export SIF_IMAGE=/path/to/osworld_sandbox
```

## Quick Start

### Training with 3SPO

```bash
# Full OSWorld training with 3SPO (2 nodes × 8 GPUs, 128 envs)
bash examples/osworld_full_3spo.sh
```

Key 3SPO hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `algorithm.adv_estimator` | `3spo` | Use 3SPO advantage estimator |
| `algorithm.three_spo_g` | `8` | Group size G for DFS branching |
| `algorithm.similarity_threshold` | `0.98` | Cosine similarity threshold for visual state grouping |
| `algorithm.lambda_base` | `1.0` | Base decay rate for state score |
| `algorithm.alpha` | `0.1` | Temporal decay factor |
| `algorithm.xi` | `8` | Failure count threshold (state pruning) |

### Training with GRPO (Baseline)

```bash
bash examples/osworld_full_grpo.sh
```

### Training with ARPO (Online Replay)

```bash
bash examples/osworld_full_arpo.sh
```

### Training on a Subset

```bash
# Quick experiment on 32 tasks
bash examples/osworld_subset32.sh
```

### Evaluation

```bash
# 1. Start the vLLM inference server
bash start_server.sh 1   # 1 GPU; change to match your setup

# 2. Run evaluation
bash examples/eval_osworld.sh
```

### Merging Checkpoints

After FSDP training, merge sharded checkpoints into a standard HuggingFace model:

```bash
python scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path /path/to/original/model \
    --local_dir /path/to/checkpoint
```

## Configuration

The project uses [OmegaConf](https://omegaconf.readthedocs.io/) for hierarchical configuration. The base config is at `examples/config.yaml`, and training scripts override specific fields via command-line arguments.

```yaml
# Key configuration groups
data:          # Dataset paths, tokenization, batch sizes
algorithm:     # RL algorithm selection and hyperparameters
worker:        # Actor, rollout, ref policy, reward settings
env:           # Desktop environment (num_envs, screen_size)
trainer:       # Training loop (episodes, logging, checkpointing)
```

### Example: Custom Training Command

```bash
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=your_tasks.json \
    data.max_prompt_length=64000 \
    data.max_response_length=8192 \
    algorithm.adv_estimator=3spo \
    algorithm.three_spo_g=8 \
    worker.actor.model.model_path=/path/to/model \
    env.num_envs=128 \
    trainer.nnodes=2 \
    trainer.n_gpus_per_node=8
```

## Project Structure

```
3SPO-GUI/
├── verl/                       # Core RL training framework
│   ├── trainer/
│   │   ├── main.py             # Entry point
│   │   ├── ray_trainer.py      # Main training loop (3SPO DFS, GRPO, etc.)
│   │   ├── core_algos.py       # RL algorithm implementations
│   │   ├── gui_agent.py        # EnvWorker (desktop environment wrapper)
│   │   ├── config.py           # Configuration dataclasses
│   │   └── replay_buffer.py    # Replay buffer for ARPO
│   ├── models/                 # VLM model patches (Qwen2-VL attention)
│   ├── workers/                # FSDP workers (actor, critic, rollout, reward)
│   ├── single_controller/      # Ray-based distributed worker management
│   └── utils/                  # Datasets, tokenizers, logging, checkpoints
├── examples/                   # Training and evaluation scripts
│   ├── config.yaml             # Base configuration
│   ├── osworld_full_3spo.sh    # 3SPO training on full OSWorld
│   ├── osworld_full_grpo.sh    # GRPO baseline training
│   ├── osworld_full_arpo.sh    # ARPO with online replay
│   ├── osworld_subset32.sh     # Subset training
│   └── eval_osworld.sh         # Evaluation script
├── scripts/                    # Utility scripts (checkpoint merging)
├── assets/                     # Documentation images
├── OSWorld/                    # Desktop environment (git submodule)
├── requirements.txt
├── setup.py
└── start_server.sh             # vLLM inference server launcher
```

## Citation

If you find 3SPO useful, please cite our work:

```bibtex
@article{3spo2025,
  title={3SPO: Step-level State Policy Optimization for GUI Grounding},
  year={2025}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

- [veRL](https://github.com/volcengine/verl) — Distributed RL training framework
- [OSWorld](https://github.com/xlang-ai/OSWorld) — Desktop environment and benchmark
- [UI-TARS](https://github.com/bytedance/UI-TARS) — Vision-language GUI agent
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — Multi-modal vision-language model
