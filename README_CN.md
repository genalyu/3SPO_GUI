# 3SPO_GUI: State-Score-Supervised Policy Optimization for GUI Agent

<p align="center">
  <a href="README.md">English</a> | <a href="#3spo-面向-gui-交互的步级状态策略优化">简体中文</a>
</p>

**3SPO_GUI**是一个面向 **GUI 智能体**的强化学习训练框架，专为在真实桌面环境中训练视觉-语言模型驱动的 GUI Agent 而设计。

与 GRPO、PPO 等**轨迹级**方法不同（仅在任务最终给出奖励），3SPO 在**每一步操作**上进行策略优化——它通过视觉状态识别来评估每一步 GUI 动作的质量，从而实现更精细的信用分配和更高效的训练。

本框架基于 [veRL](https://github.com/volcengine/verl) 和 [OSWorld](https://github.com/xlang-ai/OSWorld) 构建，采用 FSDP + vLLM 进行分布式训练，使用 Apptainer 容器化的 Ubuntu 虚拟机作为交互式桌面环境。

## 核心特性

- **3SPO 算法** — 全新的步级强化学习方法，包含视觉状态识别、自适应 Rollout 预算和 DFS 探索策略
- **真实桌面环境** — 智能体在 Ubuntu 虚拟机中通过截图和 pyautogui 指令与 Chrome、GIMP、LibreOffice、VS Code 等应用交互
- **多模态视觉-语言模型** — 原生支持 Qwen2.5-VL 和 UI-TARS 系列模型
- **分布式训练** — FSDP 模型并行 + Ray 分布式调度 + vLLM 高效推理
- **多种 RL 基线算法** — 内置 GRPO、GAE、RLOO、REINFORCE++、ReMax、ARPO
- **OSWorld 基准测试** — 覆盖 10 个桌面应用场景的 369 项任务评测

## 3SPO 算法简介

3SPO 的核心思想是将强化学习的优化粒度从**轨迹级**细化到**步级**。具体而言：

### 1. 视觉状态识别 (Visual State Identification)

通过对截图的视觉编码器特征取均值，计算状态嵌入 $\psi(s)$。使用余弦相似度阈值（默认 0.98）将视觉相似的状态归为同一组，实现跨任务的状态共享。

### 2. 步级奖励设计 (Step-level Reward)

$$r_{3SPO} = \omega \cdot r_{novel} + (0.5 - \omega) \cdot (S(s_t) - S(s_{t+1})) + 0.5 \cdot r_{osworld}$$

- **$r_{novel}$**：状态新颖度，衡量动作带来的视觉变化程度（L2 距离）
- **$S(s_t)$**：状态得分，基于历史成功率与指数衰减
- **$r_{osworld}$**：环境奖励 + 格式奖励
- **$\omega = 0.5 \cdot e^{-n_{total}}$**：探索权重，随访问次数递减

### 3. 自适应 Rollout (Adaptive Rollout)

$$n(s_t) = \lfloor G \cdot \sigma(\lambda \cdot S(s_t)) \rfloor$$

对不确定的状态（历史成功率低）分配更多采样次数，对已掌握的状态减少采样，高效利用计算资源。

### 4. DFS 探索策略

采用基于栈的 DFS 搜索遍历轨迹，在每个状态采样多个动作，选择最优子状态继续探索，并将成功/失败信息反向传播至路径上所有经过的视觉状态。

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                   RayPPOTrainer (主控节点)                     │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────────────┐  │
│  │ 3SPO DFS │  │ 状态管理器  │  │   优势值 & 奖励计算      │  │
│  │ 探索器   │  │ (ψ, VID, S)│  │  (步级 3SPO / GRPO)      │  │
│  └────┬─────┘  └────────────┘  └──────────────────────────┘  │
└───────┼──────────────────────────────────────────────────────┘
        │ Ray RPC
        ▼
┌───────────────────────────────────────────────────────────────┐
│                   Worker Groups (Ray Actors)                   │
│                                                               │
│  ┌──────────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ ActorRollout     │  │ Ref Policy   │  │ Env Workers     │  │
│  │ (FSDP + vLLM)    │  │ (FSDP)       │  │ (Ubuntu VM ×N) │  │
│  └──────────────────┘  └──────────────┘  └────────────────┘  │
└───────────────────────────────────────────────────────────────┘
```

## 支持的算法

| 算法 | 优化粒度 | 说明 |
|------|---------|------|
| **3SPO** | 步级 | 视觉状态评分 + 自适应 Rollout + DFS 探索 |
| GRPO | 轨迹级 | Group Relative Policy Optimization，组内归一化奖励 |
| GAE | 步级 | 广义优势估计（需要 Critic 网络） |
| RLOO | 轨迹级 | Leave-one-out 基线 |
| REINFORCE++ | 轨迹级 | 折扣回报 + 掩码白化 |
| ReMax | 轨迹级 | 贪心 Rollout 作为奖励基线 |
| ARPO | 轨迹级 | GRPO + 在线回放缓冲区，注入正样本 |

## 安装指南

### 环境要求

- Python >= 3.10
- CUDA >= 12.1
- Apptainer / Singularity（用于 Ubuntu 虚拟机环境）
- Ray（分布式调度）

### 安装步骤

```bash
# 克隆仓库（含子模块）
git clone --recursive https://github.com/your-org/3SPO-GUI.git
cd 3SPO-GUI

# 安装依赖
pip install -r requirements.txt
pip install -e .

# 如果未使用 --recursive 克隆，手动初始化子模块
git submodule update --init --recursive
```

### 桌面环境配置

GUI 环境在 Apptainer 容器中运行 Ubuntu 系统，需要准备：

1. **Ubuntu 虚拟机镜像**（`Ubuntu.qcow2`）— 桌面环境的磁盘镜像
2. **Apptainer 镜像 / Sandbox** — OSWorld 的容器运行环境

```bash
# （可选）如果 .sif 文件报错 "unsquashfs not found"，可解压为 Sandbox
apptainer build --sandbox osworld_sandbox osworld.sif
```

设置镜像路径（通过环境变量或直接在脚本中修改）：

```bash
export LOCAL_IMAGE=/path/to/Ubuntu.qcow2
export SIF_IMAGE=/path/to/osworld_sandbox
```

## 使用指南

### 3SPO 训练

```bash
# 完整 OSWorld 训练（2 节点 × 8 GPU，128 个并行环境）
bash examples/osworld_full_3spo.sh
```

3SPO 关键超参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `algorithm.adv_estimator` | `3spo` | 使用 3SPO 优势值估计器 |
| `algorithm.three_spo_g` | `8` | DFS 分支组大小 G |
| `algorithm.similarity_threshold` | `0.98` | 视觉状态分组的余弦相似度阈值 |
| `algorithm.lambda_base` | `1.0` | 状态得分的基础衰减率 |
| `algorithm.alpha` | `0.1` | 时间衰减因子 |
| `algorithm.xi` | `8` | 失败次数阈值（状态剪枝） |

### GRPO 基线训练

```bash
bash examples/osworld_full_grpo.sh
```

### ARPO 训练（在线回放）

```bash
bash examples/osworld_full_arpo.sh
```

### 子集训练（快速实验）

```bash
# 在 32 个任务上快速验证
bash examples/osworld_subset32.sh
```

### 模型评测

```bash
# 1. 启动 vLLM 推理服务（参数为 GPU 数量）
bash start_server.sh 1

# 2. 运行 OSWorld 评测
bash examples/eval_osworld.sh
```

### 合并模型检查点

FSDP 训练后，将分片检查点合并为标准 HuggingFace 模型：

```bash
python scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path /path/to/original/model \
    --local_dir /path/to/checkpoint
```

## 配置说明

项目使用 [OmegaConf](https://omegaconf.readthedocs.io/) 进行层级化配置管理。基础配置在 `examples/config.yaml`，训练脚本通过命令行参数覆盖特定字段。

### 配置组

| 配置组 | 说明 |
|--------|------|
| `data` | 数据集路径、分词参数、批大小 |
| `algorithm` | RL 算法选择及超参数 |
| `worker` | Actor、Rollout、Ref Policy、Reward 设置 |
| `env` | 桌面环境配置（环境数量、屏幕分辨率） |
| `trainer` | 训练循环（总轮数、日志、检查点保存） |

### 自定义训练命令示例

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

## 项目结构

```
3SPO-GUI/
├── verl/                       # 核心 RL 训练框架
│   ├── trainer/
│   │   ├── main.py             # 训练入口
│   │   ├── ray_trainer.py      # 主训练循环（3SPO DFS、GRPO 等）
│   │   ├── core_algos.py       # RL 算法实现（GAE、GRPO、3SPO 等）
│   │   ├── gui_agent.py        # EnvWorker（桌面环境封装）
│   │   ├── config.py           # 配置数据类
│   │   └── replay_buffer.py    # ARPO 回放缓冲区
│   ├── models/                 # VLM 模型补丁（Qwen2-VL 注意力机制）
│   ├── workers/                # FSDP Worker（Actor、Critic、Rollout、Reward）
│   ├── single_controller/      # Ray 分布式 Worker 管理
│   └── utils/                  # 数据集、分词器、日志、检查点管理
├── examples/                   # 训练与评测脚本
│   ├── config.yaml             # 基础配置
│   ├── osworld_full_3spo.sh    # 3SPO 全量训练
│   ├── osworld_full_grpo.sh    # GRPO 基线训练
│   ├── osworld_full_arpo.sh    # ARPO 在线回放训练
│   ├── osworld_subset32.sh     # 子集训练
│   └── eval_osworld.sh         # 评测脚本
├── scripts/                    # 工具脚本（检查点合并）
├── assets/                     # 文档图片
├── OSWorld/                    # 桌面环境（Git 子模块）
├── requirements.txt            # Python 依赖
├── setup.py                    # 包安装
└── start_server.sh             # vLLM 推理服务启动脚本
```

## 引用

如果您觉得 3SPO 对您的研究有帮助，请引用：

```bibtex
@article{3spo2025,
  title={3SPO: Step-level State Policy Optimization for GUI Grounding},
  year={2025}
}
```

## 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源协议。

## 致谢

- [veRL](https://github.com/volcengine/verl) — 分布式 RL 训练框架
- [OSWorld](https://github.com/xlang-ai/OSWorld) — 桌面环境与评测基准
- [UI-TARS](https://github.com/bytedance/UI-TARS) — 视觉-语言 GUI 智能体
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) — 多模态视觉-语言模型
