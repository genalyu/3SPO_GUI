# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""
import re
import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import copy
import numpy as np
import random
import ray
import torch
from codetiming import Timer
from ray.experimental.tqdm_ray import tqdm
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin
from collections import defaultdict

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.dataset import collate_fn as collate_fn_raw
from ..utils.osworld import OSWorldDataset, OSWorldTaskConfigDataset, OSWorldGRPODataset, collate_fn, collate_fn_dataproto, collate_fn_fake, GRPODatasetProcessor
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics

from .gui_agent import EnvWorker
from .replay_buffer import ReplayBuffer

from collections import defaultdict
from qwen_vl_utils import process_vision_info

import time
from concurrent.futures import ThreadPoolExecutor


class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"
    THREE_SPO = "3spo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}."
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    if "ref_log_probs" in data.batch.keys():
        kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
        kld = kld * response_mask  # (batch_size, response_length)
    else:
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.THREE_SPO:
        # 3SPO uses step-level scores directly passed in token_level_rewards
        # token_level_rewards: (bsz, response_length)
        scores = token_level_rewards.sum(dim=-1)
        # In 3SPO, we use the state-specific index (uid) to group for variable n(s_t)
        advantages = core_algos.compute_3spo_step_advantage(scores, response_mask, index)
        returns = advantages # 3SPO returns are just advantages for now
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last





class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
        val_reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        
        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")
        
        print(config)

        self.task_config_single = None
        self.fake_dataset = None
        self._create_dataloader()
        self._create_envs()
        self._load_replay_data()
        
        # --- 3SPO Reward Design States ---
        self.state_stats = defaultdict(lambda: {"n_success": 0, "n_total": 0, "n_fail": 0})
        self.state_embeddings = {} # (task_id, tuple(history_actions)) -> torch.Tensor (psi(s))
        
        # --- 3SPO Constants ---
        self.lambda_base = getattr(config.algorithm, "lambda_base", 1.0)
        self.alpha = getattr(config.algorithm, "alpha", 0.1)
        self.T_max = getattr(config.algorithm, "T_max", 1000) # total episodes/steps
        self.xi = getattr(config.algorithm, "xi", 3) # failure threshold
        self.epsilon = 1e-6
        self.rollout_rule_lambda = getattr(config.algorithm, "rollout_rule_lambda", 1.0)
        self.rollout_rule_G = getattr(config.algorithm, "three_spo_g", 8)
    
    def _get_state_id(self, task_id, history_actions):
        return (task_id, tuple(history_actions) if history_actions else ())

    def _compute_psi(self, env_output):
        """Compute state embedding psi(s). Use mean of pixel_values if available."""
        if 'multi_modal_inputs' in env_output and 'pixel_values' in env_output['multi_modal_inputs']:
            # pixel_values shape: [num_patches, hidden_dim] or similar
            # For Qwen2-VL it's usually [N, 1176] or [N, 10764]
            pv = env_output['multi_modal_inputs']['pixel_values']
            if isinstance(pv, torch.Tensor):
                return pv.mean(dim=0).cpu()
        # Fallback to zero vector if no vision info
        return torch.zeros(128)

    def _get_lambda_t(self, t):
        return self.lambda_base * (1 - np.exp(-self.alpha * t / self.T_max))

    def _get_state_score(self, state_id, t):
        stats = self.state_stats[state_id]
        if stats["n_fail"] >= self.xi:
            return 0.0
        
        lambda_t = self._get_lambda_t(t)
        success_rate = stats["n_success"] / (stats["n_total"] + self.epsilon)
        score = np.exp(-lambda_t * success_rate)
        return float(score)

    def _get_omega(self, n_total):
        return 0.5 * np.exp(-n_total)

    def _compute_r_novel(self, psi_t, psi_next):
        if psi_t is None or psi_next is None:
            return 0.0
        diff_norm = torch.norm(psi_next - psi_t, p=2)
        base_norm = torch.norm(psi_t, p=2)
        return float(diff_norm / (base_norm + self.epsilon))

    def _compute_r_3spo(self, state_id, next_state_id, r_osworld, t):
        psi_t = self.state_embeddings.get(state_id)
        psi_next = self.state_embeddings.get(next_state_id)
        
        n_total = self.state_stats[state_id]["n_total"]
        omega = self._get_omega(n_total)
        
        r_novel = self._compute_r_novel(psi_t, psi_next)
        
        s_t = self._get_state_score(state_id, t)
        s_next = self._get_state_score(next_state_id, t)
        
        r_3spo = omega * r_novel + (0.5 - omega) * (s_t - s_next) + 0.5 * r_osworld
        return r_3spo

    def _compute_adaptive_n(self, state_id, t, entropy):
        s_t = self._get_state_score(state_id, t)
        # n(s_t) = floor(G * Sigmoid(lambda * S(s_t) * H))
        val = self.rollout_rule_lambda * s_t * entropy
        sigmoid_val = 1.0 / (1.0 + np.exp(-val))
        n = int(np.floor(self.rollout_rule_G * sigmoid_val))
        return max(1, n) # Ensure at least 1 rollout

    def _update_stats_backprop(self, task_id, history_actions, is_success):
        """Backpropagate success/fail info up the history tree."""
        for i in range(len(history_actions) + 1):
            sub_history = history_actions[:i]
            sid = self._get_state_id(task_id, sub_history)
            self.state_stats[sid]["n_total"] += 1
            if is_success:
                self.state_stats[sid]["n_success"] += 1
            else:
                self.state_stats[sid]["n_fail"] += 1

    def _load_replay_data(self):
        data_path = None
        self.replay = ReplayBuffer(data_path, 8)



    def _create_envs(self) -> None:
        """
        Create env workers and data-processor workers, 
        and pin each EnvWorker to a different node (round-robin).
        """
        print('Start to create env_worker for OSWorld Environment')
        max_steps = self.config.env.max_steps
        num_envs = self.config.env.num_envs

        # 1) 从 cluster_resources 里挑出自定义的 IP 资源标签
        #    cluster_resources() 里还会有 "CPU"/"GPU"/"memory" 等内置资源，我们要过滤掉
        all_res = ray.cluster_resources().keys()
        # ip_labels = [r for r in all_res if re.match(r"^\d+\.\d+\.\d+\.\d+$", r)]
        ip_labels = [r for r in all_res if re.match(r"^docker:\d+\.\d+\.\d+\.\d+$", r)]
        if not ip_labels:
            raise RuntimeError("没找到任何 IP 资源标签，请检查 ray start 时 --resources 参数")

        # 2) 按 round-robin 方式，把每个 env worker pin 到不同节点
        self.env_workers = []
        for i in range(num_envs):
            ip_label = ip_labels[i % len(ip_labels)]
            w = EnvWorker.options(
                    resources={ ip_label: 1 },   # 保证这个 actor 一定被调度到拥有 ip_label 资源的节点
                    name=f"env_worker_{i}"
                ).remote(i, max_steps, self.config)
            self.env_workers.append(w)

        print(f'Env_worker for OSWorld Environment created!  total: {len(self.env_workers)}')

        # 3) 数据预处理器，放在 driver 或随意放一个节点上都行
        self.data_processor_workers = [
            GRPODatasetProcessor.remote(
                self.processor,
                self.tokenizer,
                max_prompt_length=self.config.data.max_prompt_length
            )
            for _ in range(num_envs)
        ] 
            
    def _create_dataloader(self) -> None:
        self.train_dataset = OSWorldTaskConfigDataset(
            data_path=self.config.data.train_files,
        )
        # data = self.train_dataset[0]
        # breakpoint()
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        self.val_dataset = OSWorldTaskConfigDataset(
            data_path=self.config.data.val_files,
        )
        
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=min(self.config.env.num_envs, len(self.val_dataset)), # use the same number as envs
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")


        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)

        task_configs_total = []
        eval_results_total = []
        for batch_dict in self.val_dataloader:
            task_configs = batch_dict
            num_tasks = len(task_configs)
            assert num_tasks <= self.config.env.num_envs
            task_configs_total.extend(task_configs) # record task

            futures = [
                worker.reset.remote(task_config) for worker, task_config in
                zip(self.env_workers[:num_tasks], task_configs)
            ]
            reset_outputs = ray.get(futures)

            self.actor_rollout_wg.prepare_generate_sequences()

            env_outputs = reset_outputs

            for step_idx in range(self.config.env.max_steps):
                print(f"Step {step_idx} of {self.config.env.max_steps}: {ray.get([worker.is_done.remote() for worker in self.env_workers])}")
                num_workers = len(self.env_workers)

                vllm_batch, valid_env_idx = self.prepare_vllm_inputs_full(env_outputs)

                vllm_batch_pad, pad_size = pad_dataproto_to_divisor(vllm_batch, num_workers)
                
                gen_batch = vllm_batch_pad.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                )

                # override val config
                gen_batch.meta_info = self.config.worker.rollout.val_override_config

                # predict actions
                action_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                action_batch_output = unpad_dataproto(action_batch_output, pad_size=pad_size)
                
                response_texts = self.tokenizer.batch_decode(action_batch_output.batch['responses'], skip_special_tokens=True)

                cur_valid_envs = [self.env_workers[i] for i in valid_env_idx]

                futures = [worker.step.remote(action_text) for worker, action_text in zip(cur_valid_envs, response_texts)]
                env_outputs = ray.get(futures)

                is_all_done = all([x['is_done'] for x in env_outputs])
                if is_all_done:
                    break

            futures = [worker.evaluate.remote() for worker in self.env_workers[:num_tasks]]
            eval_results = ray.get(futures)
            eval_results_total.extend(eval_results)

            history_messages = ray.get([worker.get_history_messages.remote() for worker in self.env_workers[:num_tasks]])
            self.actor_rollout_wg.finish_generate_sequences()

            # Store scores
            scores = eval_results
            reward_tensor = torch.tensor(scores, dtype=torch.float32).unsqueeze(-1)

            sample_inputs.extend([task_config['instruction'] for task_config in task_configs])
            prompts = []
            for history_message in history_messages:
                prompts.append(self.processor.apply_chat_template(history_message))
            
            sample_outputs.extend(prompts)
            sample_labels.extend(['none']*len(prompts))
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)

        # Store eval_results
        save_path = os.path.join(self.config.trainer.save_checkpoint_path, f"eval_results_at_{self.global_step}.json")
        save_dict = dict()
        for task_config, eval_result in zip(task_configs_total, eval_results_total):
            task_id = task_config['task_id']
            save_dict[task_id] = eval_result

        if not os.path.exists(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(save_dict, f, indent=4)

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        return {"val/reward_score": reward_score}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    
    def prepare_vllm_inputs_full(self, env_outputs: List):
        # NOTE: processor will be very slow
        obs_messages = [x['obs_messages'] for x in env_outputs]
        env_idx = [x['env_idx'] for x in env_outputs]

        valid_obs_messages = [x['obs_messages'] for x in env_outputs if x['obs_messages'] is not None]
        valid_env_idx = [x['env_idx'] for x in env_outputs if x['obs_messages'] is not None]

        dataset = OSWorldDataset(
            valid_obs_messages,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            format_prompt=self.config.data.format_prompt,
            max_pixels=self.config.data.max_pixels,
            min_pixels=self.config.data.min_pixels,
            fast_rollout=True,
        )

        # batch_dict = [dataset[i] for i in range(len(dataset))]
        def get_dataset_item(index):
            return dataset[index]

        with ThreadPoolExecutor(max_workers=64) as executor:
            batch_dict = list(executor.map(get_dataset_item, range(len(dataset))))

        # batch_dict = ray.get([get_dataset_item.remote(i) for i in range(len(dataset))])

        batch_dict = collate_fn_dataproto(batch_dict)
        batch = DataProto.from_single_dict(batch_dict)
        
        return batch, valid_env_idx


    def prepare_grpo_inputs(self, messages, eval_results, task_configs):
        eval_result_flatten = eval_results
        messages_flatten = messages

        dataset = OSWorldGRPODataset(
            messages_flatten,
            tokenizer=self.tokenizer,
            processor=self.processor,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            format_prompt=self.config.data.format_prompt,
            max_pixels=self.config.data.max_pixels,
            min_pixels=self.config.data.min_pixels,
        )
        def get_dataset_item(index):
            return dataset[index]

        with ThreadPoolExecutor(max_workers=64) as executor:
            batch_dict = list(executor.map(get_dataset_item, range(len(dataset))))
        # batch_dict = [get_dataset_item(i) for i in range(len(dataset))]
        
        batch_dict = collate_fn_dataproto(batch_dict)
        batch = DataProto.from_single_dict(batch_dict)

        # uid
        # use batch to compute norm reward
        batch.non_tensor_batch["uid"] = np.array([x['id'] for x in task_configs], dtype=object)
        batch.non_tensor_batch["task_id"] = np.array([x['id'] for x in task_configs], dtype=object)

        batch.batch["rewards"] = torch.tensor([float(x) for x in eval_result_flatten], dtype=torch.float32)

        return batch


            

    def save_rollout_trajectories(self, action_batch_output, history_messages, eval_results, task_configs):
        visual_trajs = dict()
        visual_trajs['history_messages'] = history_messages
        visual_trajs['eval_results'] = eval_results
        visual_trajs['task_configs'] = task_configs
    
        # os.makedirs(self.config.trainer.save_checkpoint_path, exist_ok=True)
        os.makedirs(os.path.join(self.config.trainer.save_checkpoint_path, "trajs"), exist_ok=True)
        visual_folder_path = os.path.join(self.config.trainer.save_checkpoint_path, "trajs", f"global_step_{self.global_step}.pth")
        torch.save(visual_trajs, visual_folder_path)
        action_batch_output.save_to_disk(os.path.join(self.config.trainer.save_checkpoint_path, "trajs", f"global_step_{self.global_step}_batch.pkl"))

    def start_reset_envs(self, batch_dict):
        rollout_n = self.config.worker.rollout.n
        num_envs = self.config.env.num_envs
        num_groups = num_envs // rollout_n

        reset_envs_object = []
        task_configs = []

        for i in range(num_groups):
            # Try to get a state from ReplayBuffer for ARPO Replay
            use_replay = False
            if self.config.algorithm.enable_replay and self.config.algorithm.adv_estimator != AdvantageEstimator.THREE_SPO:
                # ARPO uses a random replay strategy to provide positive samples
                if random.random() < 0.3:
                    task_config = batch_dict[i % len(batch_dict)]
                    task_id = task_config['id']
                    pos_batch = self.replay.get_pos(task_id, num_samples=1)
                    if len(pos_batch) > 0 and 'history_actions' in pos_batch.non_tensor_batch:
                        history_actions = pos_batch.non_tensor_batch['history_actions'][0]
                        print(f"Group {i}: Continuing from ReplayBuffer for task {task_id} with {len(history_actions)} actions.")
                        for _ in range(rollout_n):
                            task_configs.append(task_config)
                            worker = self.env_workers[len(reset_envs_object)]
                            reset_envs_object.append(worker.reset_to_state.remote(task_config, history_actions))
                        use_replay = True

            if not use_replay:
                task_config = batch_dict[i % len(batch_dict)]
                for _ in range(rollout_n):
                    task_configs.append(task_config)
                    worker = self.env_workers[len(reset_envs_object)]
                    reset_envs_object.append(worker.reset.remote(task_config))

        assert len(task_configs) == len(self.env_workers)
        return task_configs, reset_envs_object
    
    def apply_replay(self, task_configs, batch):
        eval_results = batch.batch["eval_results"].tolist()
        assert len(task_configs) == len(batch)

        rollout_n = self.config.worker.rollout.n
        bsz = len(task_configs) // rollout_n

        final_batch = []
        final_eval_results = []
        for i in range(bsz):
            cur_task_config = task_configs[i * rollout_n:(i + 1) * rollout_n]
            assert len(set([x['id'] for x in cur_task_config])) == 1
            task_id = cur_task_config[0]['id']
            instruction = cur_task_config[0]['instruction']

            cur_eval_results = eval_results[i * rollout_n:(i + 1) * rollout_n]

            cur_rewards = np.array(eval_results[i * rollout_n:(i + 1) * rollout_n], dtype=float)
            cur_batch = batch[i * rollout_n:(i + 1) * rollout_n]
            cur_reward_std = np.std(cur_rewards)
            cur_reward_mean = np.mean(cur_rewards)
            if cur_reward_std < 0.05 and cur_reward_mean < 0.2: # all negative group
                pos_batch = self.replay.get_pos(cur_task_config[0]['id'], num_samples=1)
            else:
                pos_batch = []

            if len(pos_batch) > 0:
                final_batch.append(pos_batch)
                final_batch.append(cur_batch[len(pos_batch):])
            else:
                final_batch.append(cur_batch)

            print(f'Task {task_id} {instruction} replay_buffer: {len(pos_batch)}| rewards: {cur_rewards}')
            # print(f'len(final_messages): {len(final_messages)}, len(final_eval_results): {len(final_eval_results)}')
        
        # update replay buffer
        self.replay.update_replay_buffer_batch(task_configs, batch)
        print('Update replay buffer done')
        final_batch = DataProto.concat(final_batch)
        return final_batch

    def run_3spo_rollout_step(self, step_idx, task_configs, active_workers):
        """Perform a single rollout step for a specific subset of workers."""
        # 1. Get observations only from active workers
        env_outputs = ray.get([worker.get_obs.remote() for worker in active_workers])
        
        # 2. VLLM Inference
        vllm_batch, valid_env_idx = self.prepare_vllm_inputs_full(env_outputs)
        num_workers = len(self.actor_rollout_wg._workers)
        vllm_batch_pad, pad_size = pad_dataproto_to_divisor(vllm_batch, num_workers)
        
        gen_batch = vllm_batch_pad.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
        )
        
        action_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
        action_batch_output = unpad_dataproto(action_batch_output, pad_size=pad_size)
        
        response_texts = self.tokenizer.batch_decode(action_batch_output.batch['responses'], skip_special_tokens=True)
        
        # 3. Environment Step for active workers
        futures = [
            worker.step.remote(action_text) for worker, action_text in zip(active_workers, response_texts)
        ]
        new_env_outputs = ray.get(futures)
        
        # Add task_config back to outputs for tracking
        for out, cfg in zip(new_env_outputs, task_configs):
            out['task_config'] = cfg
            
        return new_env_outputs

    def run_3spo_step(self, step_idx, env_outputs, task_configs, active_workers, state_ids, timing_raw):
        """Runs a single 3SPO step: sample G actions, compute advantages, update model, pick best."""
        # state_ids is a list of state identifiers for each output in env_outputs
        
        # 2. Collect rewards and training data only from active workers
        process_results = ray.get([worker.get_step_train_dict.remote(step_idx) for worker in active_workers])
        batch = collate_fn_dataproto(process_results)
        batch = DataProto.from_single_dict(batch)

        # 3. Compute scores using the new 3SPO reward design
        scores = []
        for i, out in enumerate(env_outputs):
            task_id = out['task_config']['id']
            history_actions = out['history_actions']
            parent_history = history_actions[:-1]
            
            sid = self._get_state_id(task_id, parent_history)
            next_sid = self._get_state_id(task_id, history_actions)
            
            # Update embeddings if not present
            if sid not in self.state_embeddings:
                self.state_embeddings[sid] = self._compute_psi(out) # Placeholder
            
            if next_sid not in self.state_embeddings:
                self.state_embeddings[next_sid] = self._compute_psi(out)
                
            r_osworld = float(out.get('eval_result', 0)) + 0.5 * float(out['format_reward'])
            r_3spo = self._compute_r_3spo(sid, next_sid, r_osworld, self.global_step)
            scores.append(r_3spo)
            
            # If child is terminal, backpropagate success/fail
            if out['is_done']:
                is_success = float(out.get('eval_result', 0)) > 0.05
                self._update_stats_backprop(task_id, history_actions, is_success)
            else:
                # If not terminal, we still count it as a "total" visit for the parent
                self.state_stats[sid]["n_total"] += 1

        scores = torch.tensor(scores, dtype=torch.float32)
        
        batch.batch["token_level_rewards"] = scores.unsqueeze(-1)
        batch.batch["responses"] = batch.batch["input_ids"]
        batch.batch["response_mask"] = batch.batch["labels"]!=-100
        batch.batch["eval_results"] = torch.tensor([float(out.get('eval_result', 0)) for out in env_outputs], dtype=torch.float32)
        batch.non_tensor_batch["uid"] = np.array(state_ids, dtype=object) # Use state_ids to group correctly
        batch.non_tensor_batch["task_id"] = np.array([x['id'] for x in task_configs], dtype=object)

        # 4. Compute advantages (This correctly groups by G internally)
        batch = compute_advantage(batch, adv_estimator=AdvantageEstimator.THREE_SPO)

        # 5. Update Actor
        with _timer("update_actor_3spo", timing_raw):
            actor_output = self.actor_rollout_wg.update_actor(batch)

        # 6. Save to ReplayBuffer
        history_actions_list = [out.get('history_actions', None) for out in env_outputs]
        self.replay.update_replay_buffer_batch(task_configs, batch, history_actions_list=history_actions_list)

        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        rollout_n = self.config.worker.rollout.n
        for _ in tqdm(range(self.config.trainer.total_episodes), desc="Episode", position=0):
            # --- 3SPO DFS Mode ---
            if self.config.algorithm.adv_estimator == AdvantageEstimator.THREE_SPO:
                G = rollout_n
                K = len(self.env_workers) // G
                stacks = [[] for _ in range(K)]
                train_iter = iter(self.train_dataloader)
                
                # Fill stacks with initial tasks
                try:
                    batch_dict = next(train_iter)
                    for i, cfg in enumerate(batch_dict):
                        stacks[i % K].append({"config": cfg, "actions": [], "depth": 0})
                except StopIteration:
                    break
                
                while True:
                    active_parents = [None] * K
                    active_slots = []
                    
                    for k in range(K):
                        # If this slot's stack is empty, try to refill it with a new task
                        if not stacks[k]:
                            try:
                                batch_dict = next(train_iter)
                                # Distribute the new batch of tasks among empty stacks
                                for cfg in batch_dict:
                                    # Find the first empty stack to put this task
                                    for target_k in range(K):
                                        if not stacks[target_k] and target_k not in active_slots:
                                            stacks[target_k].append({"config": cfg, "actions": [], "depth": 0})
                                            break
                            except StopIteration:
                                pass # No more tasks in dataloader

                        # If it now has a task, pick it up
                        if stacks[k]:
                            active_parents[k] = stacks[k].pop()
                            active_slots.append(k)
                    
                    if not active_slots:
                        break
                        
                    # Prepare task configs for workers
                    round_task_configs = []
                    reset_objects = []
                    active_workers = []
                    round_state_ids = []
                    
                    for k in active_slots:
                        parent = active_parents[k]
                        task_id = parent["config"]["id"]
                        sid = self._get_state_id(task_id, parent["actions"])
                        
                        # Calculate adaptive n(s_t) BEFORE rollout
                        # Default entropy=1.0 for now
                        nk = self._compute_adaptive_n(sid, self.global_step, entropy=1.0)
                        
                        # Each slot k uses its dedicated group of G workers: [k*G, (k+1)*G)
                        # But we only use the first nk workers
                        for g_idx in range(nk):
                            round_task_configs.append(parent["config"])
                            worker_idx = k * G + g_idx
                            worker = self.env_workers[worker_idx]
                            active_workers.append(worker)
                            round_state_ids.append(sid)
                            reset_objects.append(worker.reset_to_state.remote(parent["config"], parent["actions"]))
                    
                    # Execute expansion round
                    ray.get(reset_objects)
                    
                    # One-step rollout for all envs in this round
                    # Use the depth of the first parent for logging/step_idx
                    self.global_step += 1
                    current_depth = active_parents[active_slots[0]]["depth"]
                    env_outputs = self.run_3spo_rollout_step(current_depth, round_task_configs, active_workers)
                    
                    # 3. Step-level Opt
                    step_batch = self.run_3spo_step(current_depth, env_outputs, round_task_configs, active_workers, round_state_ids, {})
                    
                    # 4. Collect results and push back to stacks for next step (DFS)
                    # We need to correctly slice env_outputs based on nk
                    output_idx = 0
                    for k in active_slots:
                        parent = active_parents[k]
                        task_id = parent["config"]["id"]
                        sid = self._get_state_id(task_id, parent["actions"])
                        nk = self._compute_adaptive_n(sid, self.global_step, entropy=1.0)
                        
                        slot_outputs = env_outputs[output_idx : output_idx + nk]
                        output_idx += nk
                        
                        # Sort by score using the new 3SPO reward design
                        scored_children = []
                        for out in slot_outputs:
                            # score is already computed inside run_3spo_step or we can recompute
                            # Actually, let's just recompute for sorting
                            history_actions = out['history_actions']
                            next_sid = self._get_state_id(task_id, history_actions)
                            r_osworld = float(out.get('eval_result', 0)) + 0.5 * float(out['format_reward'])
                            score = self._compute_r_3spo(sid, next_sid, r_osworld, self.global_step)
                            scored_children.append((score, out))
                        
                        # Sort ascending so the best one is at the end (popped last)
                        scored_children.sort(key=lambda x: x[0])
                        
                        # Push all generated children to stack (since we already limited to nk)
                        for score, out in scored_children:
                            if not out['is_done'] and (parent["depth"] + 1) < self.config.env.max_steps:
                                stacks[k].append({
                                    "config": out['task_config'],
                                    "actions": out['history_actions'],
                                    "depth": parent["depth"] + 1
                                })
                continue # Skip the normal PPO loop
            
            # --- Normal PPO/ARPO Mode ---
            for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=1):
                self.global_step += 1
                # if self.global_step > self.training_steps or batch_dict_next_batch is None:
                if self.global_step > self.training_steps:
                    break

                # batch_dict = batch_dict_next_batch
                # task_configs, reset_envs_object = task_configs_next_batch, reset_envs_object_next_batch

                task_configs, reset_envs_object = self.start_reset_envs(batch_dict)

                metrics, timing_raw = {}, {}


                print([config['id'] for config in task_configs])
                print(f"task_num: {len(task_configs)}, env_num: {len(self.env_workers)}")
                print([config['instruction'] for config in task_configs])

                with _timer("step", timing_raw):
                    self.actor_rollout_wg.prepare_generate_sequences()

                    assert len(task_configs) == len(self.env_workers)

                    # generate a batch
                    format_rewards = [0.] * len(task_configs)
                    eval_results_objects = [None] * len(task_configs)

                    with _timer(f"gen", timing_raw):  # wg: worker group

                        with _timer("env_reset", timing_raw):
                            # reset_outputs = ray.get([
                            #     worker.reset.remote(task_config) for worker, task_config in 
                            #     zip(self.env_workers, cur_task_configs)
                            # ])
                            reset_outputs = ray.get(reset_envs_object)
                            
                        print(f"reset_time: {timing_raw['env_reset']}")

                        env_outputs = reset_outputs
                        for step_idx in range(self.config.env.max_steps):
                            is_done_stats = ray.get([worker.is_done.remote() for worker in self.env_workers])
                            print(f'step_idx: {step_idx}, finished: {sum(is_done_stats)}')

                            num_workers = len(self.actor_rollout_wg._workers)
                            with _timer("prepare_vllm_inputs", timing_raw):
                                vllm_batch, valid_env_idx = self.prepare_vllm_inputs_full(env_outputs)

                            print('prepare_vllm_inputs_time: ', timing_raw['prepare_vllm_inputs'])
                            vllm_batch_pad, pad_size = pad_dataproto_to_divisor(vllm_batch, num_workers)

                            gen_batch = vllm_batch_pad.pop(
                                batch_keys=["input_ids", "attention_mask", "position_ids"],
                                non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"],
                            )
                            # predict actions
                            with _timer("actor_rollout_wg", timing_raw):
                                action_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            print('action_batch_output_time: ', timing_raw['actor_rollout_wg'])
                            action_batch_output = unpad_dataproto(action_batch_output, pad_size=pad_size)

                            response_texts = self.tokenizer.batch_decode(action_batch_output.batch['responses'], skip_special_tokens=True)

                            cur_valid_envs = [self.env_workers[i] for i in valid_env_idx]
                            with _timer("env_step", timing_raw):
                                futures = [
                                    worker.step.remote(action_text) for worker, action_text in zip(cur_valid_envs, response_texts)
                                ]
                                env_outputs = ray.get(futures)
                            print('env_step_time: ', timing_raw['env_step'])

                            # get format rewards
                            for single_output in env_outputs:
                                if single_output['is_done']:
                                    cur_env_idx = single_output['env_idx']
                                    format_rewards[cur_env_idx] = single_output['format_reward']
                                    # start evaluate, do not evaluate in the end together
                                    eval_results_objects[cur_env_idx] = self.env_workers[cur_env_idx].evaluate.remote()

                            is_all_done = all([x['is_done'] for x in env_outputs])
                            if is_all_done:
                                break

                        # history_messages = ray.get([worker.get_history_messages.remote() for worker in self.env_workers])

                        # start evaluation
                        # eval_results = [worker.evaluate.remote() for worker in self.env_workers]
                        assert None not in eval_results_objects, 'eval_results_objects should not be None'

                        # if self.global_step % 1 == 0:
                            # self.save_rollout_trajectories(action_batch_output, history_messages, eval_results, task_configs)
                                
                    self.actor_rollout_wg.finish_generate_sequences()

                    with _timer("evaluate_env", timing_raw):
                        eval_results = ray.get(eval_results_objects)
                        # eval_results = ray.get(eval_results)
                    print('evaluate_env_time: ', timing_raw['evaluate_env'])
                    
                    with _timer("prepare_grpo_inputs", timing_raw):
                        process_results = ray.get([worker.get_train_dict.remote() for worker in self.env_workers])
                        batch = collate_fn_dataproto(process_results)
                        batch = DataProto.from_single_dict(batch)

                        batch.batch["eval_results"] = torch.tensor([float(x) for x in eval_results], dtype=torch.float32)
                        batch.batch["format_rewards"] = torch.tensor([float(x) for x in format_rewards], dtype=torch.float32)
                        batch.non_tensor_batch["uid"] = np.array([x['id'] for x in task_configs], dtype=object)
                        batch.non_tensor_batch["task_id"] = np.array([x['id'] for x in task_configs], dtype=object)
                        
                    
                    with _timer("replay", timing_raw):
                        if self.config.algorithm.enable_replay:
                            batch = self.apply_replay(task_configs, batch)

                    batch.batch["responses"] = batch.batch["input_ids"]
                    batch.batch["response_mask"] = batch.batch["labels"]!=-100

                    print('prepare_grpo_inputs_time: ', timing_raw['prepare_grpo_inputs'], '| batch size: ', len(batch))

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()


                    # compute reward
                    with _timer("reward", timing_raw):
                        # self.save_rollout_trajectories(action_batch_output, history_messages_global, eval_results_global, task_conf
                        rewards = batch.batch["eval_results"] + 0.5 * batch.batch["format_rewards"]
                        batch.batch["rewards"] = rewards

                        if self.use_reward_model:
                            raise NotImplementedError("Reward model is not supported yet.")

                        task_id_set = set(batch.non_tensor_batch["task_id"])
                        valid_task_id_set = set()

                        reward_stds_list = []
                        for task_id in task_id_set:
                            reward_in_group = batch.batch["rewards"][batch.non_tensor_batch["task_id"] == task_id]
                            # compute std in gruop
                            reward_std = reward_in_group.std().item()
                            reward_stds_list.append(reward_std)
                        
                        
                        num_invalid_group = len([x_std for x_std in reward_stds_list if x_std < 0.01])
                        print(f"num_invalid_group: {num_invalid_group}/{len(reward_stds_list)} | reward_stds_list: {reward_stds_list}")

                        # we combine with rule-based rm
                        reward_tensor = batch.batch["rewards"]
                        reward_metrics = {
                            "reward_tensor": reward_tensor.tolist(),
                            "reward_std": reward_stds_list,
                            'num_invalid_group': num_invalid_group,
                            'traj_reward': eval_results,
                            'foramt_reward': format_rewards,
                        }

                        batch.batch["token_level_scores"] = reward_tensor.unsqueeze(-1)
                        reward_metrics = {
                            f"reward/{key}": value for key, value in reduce_metrics(reward_metrics).items()
                        }
                        metrics.update(reward_metrics)
                    
                        eval_results_global_np = batch.batch["eval_results"].reshape(-1, rollout_n)
                        format_rewards_np = batch.batch["format_rewards"].reshape(-1, rollout_n)
                        print(f'Evaluation results:\n{eval_results_global_np}\nFormat rewards:\n{format_rewards_np}')
                        print('Global eval_results: ', sum(reward_tensor.tolist())/len(batch))
                    

                    # recompute old_log_probs
                    with _timer("old", timing_raw):
                        old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                        batch = batch.union(old_log_probs)

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)

                    # reset the envs for next batch
                    # is_validate_step = (
                    #     self.config.trainer.val_freq > 0
                    #     and self.global_step % self.config.trainer.val_freq == 0
                    # )
                    # try:
                    #     batch_dict_next_batch = next(iterator)
                        
                    #     if not is_validate_step:
                    #         # if is_validate_step, we will reset the envs after validation
                    #         task_configs_next_batch, reset_envs_object_next_batch = self.start_reset_envs(batch_dict_next_batch)
                    # except StopIteration:
                    #     batch_dict_next_batch = None

                    # compute ref_log_probs
                    if self.use_reference_policy:
                        with _timer("ref", timing_raw):
                            ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                            batch = batch.union(ref_log_probs)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # apply kl penalty if available
                        if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                            # apply kl penalty to reward
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)

                        critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
                        metrics.update(critic_metrics)

                    # update actor
                    if self.config.algorithm.adv_estimator != AdvantageEstimator.THREE_SPO:
                        if self.config.trainer.critic_warmup <= self.global_step:
                            with _timer("update_actor", timing_raw):
                                actor_output = self.actor_rollout_wg.update_actor(batch)

                            actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
                            metrics.update(actor_metrics)

                    # validate
                    if (
                        self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                    ):
                        with _timer("validation", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)
                        # reset the envs after validation
                        # task_configs_next_batch, reset_envs_object_next_batch = self.start_reset_envs(batch_dict_next_batch)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                self.logger.log(data=metrics, step=self.global_step)

        # perform validation after training
        if (
            val_metrics is None
            or self.config.trainer.val_freq <= 0
            or self.global_step % self.config.trainer.val_freq != 0
        ):
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)

        print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
            self._save_checkpoint()
