# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from trl import GRPOConfig, ModelConfig, TrlParser

try:
    import swanlab

    _swanlab_available = True
except ImportError:
    _swanlab_available = False

from omni_r1.debug_utils import wait_for_debugger_if_rank0
from omni_r1.load_datasets import (
    REFAVS,
    SegActDataset,
    SegZeroDataset,
    VideoMeVISDataset,
    VideoR1Dataset,
    VideoReVOSDataset,
)
from omni_r1.reward import accuracy_reward, format_reward
from omni_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer
from omni_r1.trainer.sampler import GroupSampler


@dataclass
class GRPOScriptArguments:
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={
            "help": "List of reward functions. Possible values: 'accuracy', 'format'"
        },
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )
    len_control: Optional[bool] = field(
        default=True,
        metadata={"help": "whether using length reward"},
    )
    datasets_json: Optional[str] = field(
        default="datasets.json",
        metadata={"help": "the path_name of the json specifying datasets' path"},
    )
    training_datasets: list[str] = field(
        default_factory=lambda: ["ReVOS"],
        metadata={"help": "the datasets used for training"},
    )
    use_prompt_forcing: Optional[bool] = field(
        default=False,
        metadata={"help": "whether using prompt forcing for VOS dataset"},
    )
    use_multi_vos: Optional[bool] = field(
        default=False,
        metadata={"help": "whether re-using multi-obj for VOS dataset"},
    )
    use_avs_sample_number: Optional[int] = field(
        default=1600,
        metadata={
            "help": "the number of samples to use for AVS dataset, 0 means using all samples"
        },
    )
    alpha_k: Optional[float] = field(
        default=1.0,
        metadata={"help": "the weight of R_k reward"},
    )
    alpha_k_ratio: Optional[float] = field(
        default=1.0,
        metadata={"help": "the weight of mask sum quality in R_k reward"},
    )
    alpha_a: Optional[float] = field(
        default=1.0,
        metadata={"help": "the weight of R_a reward"},
    )
    alpha_g: Optional[float] = field(
        default=1.0,
        metadata={"help": "the weight of R_g reward"},
    )


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


training_datasets = [
    "SegZero",
    "VideoR1",
    "ReVOS",
    "MeVIS",
    "RefAVS",
]


def main(script_args, training_args, model_args):
    reward_funcs = []
    # Get reward functions
    for func in script_args.reward_funcs:
        if func == "accuracy":
            partial_func = partial(
                reward_funcs_registry[func],
                log_dir=training_args.output_dir,
                alpha_a=script_args.alpha_a,
                alpha_k=script_args.alpha_k,
                alpha_g=script_args.alpha_g,
                alpha_k_ratio=script_args.alpha_k_ratio,
            )
        else:
            partial_func = reward_funcs_registry[func]
        reward_funcs.append(partial_func)

    with open(script_args.datasets_json, "r") as file:
        datasets = json.load(file)

    # Initialize a combined dataset
    combined_dataset = []
    supplementary_dataset = []
    training_datasets = script_args.training_datasets

    # Load and combine datasets based on training_datasets list
    for dataset_name in training_datasets:
        try:
            if dataset_name == "VideoR1":
                dataset = VideoR1Dataset(dataset_path=datasets.get(dataset_name, ""))
            elif dataset_name == "SegZero":
                dataset = SegZeroDataset(dataset_path=datasets.get(dataset_name, ""))
            elif dataset_name == "ReVOS":
                dataset = VideoReVOSDataset(
                    dataset_path=datasets.get(dataset_name, ""),
                    use_prompt_forcing=script_args.use_prompt_forcing,
                    select_number=3,
                    min_frames=16,
                )
                if script_args.use_multi_vos:
                    supplementary_dataset.append(
                        dataset.get_supplementary_dataset(
                            "revos_supplementary_train_.json"
                        )
                    )
            elif dataset_name == "MeVIS":
                dataset = VideoMeVISDataset(
                    dataset_path=datasets.get(dataset_name, ""),
                    use_prompt_forcing=script_args.use_prompt_forcing,
                    select_number=3,
                    min_frames=16,
                )
                if script_args.use_multi_vos:
                    supplementary_dataset.append(
                        dataset.get_supplementary_dataset(
                            "mevis_supplementary_train_.json"
                        )
                    )
            elif dataset_name == "RefAVS":
                dataset = REFAVS(
                    dataset_path=datasets.get(dataset_name, ""),
                    pick=script_args.use_avs_sample_number,
                )
            elif dataset_name == "RefCOCOg":
                dataset = SegActDataset(dataset_path=datasets.get(dataset_name, ""))
            else:
                print(f"Warning: Unknown dataset {dataset_name}, skipping")
                continue
        except Exception as e:
            raise f"Preparing datasets Error: {e}"
        combined_dataset.append(dataset)

    combined_dataset = ConcatDataset(combined_dataset + supplementary_dataset)

    assert combined_dataset is not None, "No valid datasets found for training."

    # Use the combined dataset for training
    dataset = combined_dataset

    dataset_sources = []
    dataset_indices = {}

    # 跟踪每个样本来自哪个数据集
    cumulative_size = 0
    dataset_idx = 0

    for ds in combined_dataset.datasets:
        ds_size = len(ds)
        dataset_sources.extend([dataset_idx] * ds_size)
        dataset_indices[dataset_idx] = (cumulative_size, cumulative_size + ds_size)
        cumulative_size += ds_size
        dataset_idx += 1

    # 定义组映射 - 将 RefAVS 设为组 0，其他设为组 1
    group_ids = []
    for ds in combined_dataset.datasets:
        if isinstance(ds, REFAVS):
            group_ids.append(0)  # RefAVS 组
        else:
            group_ids.append(1)  # 其他组
    # 创建自定义采样器
    is_distributed = True
    if is_distributed:
        num_replicas = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    else:
        num_replicas = 1
        rank = 0

    sampler = GroupSampler(
        data_sources=dataset_sources,
        group_ids=group_ids,
        batch_size=training_args.per_device_train_batch_size,
        shuffle=True,
        seed=training_args.seed,
        distributed=is_distributed,
        num_replicas=num_replicas,
        rank=rank,
    )

    # 创建带有自定义采样器的数据加载器
    data_loader = DataLoader(
        dataset,
        batch_size=training_args.per_device_train_batch_size,
        sampler=sampler,
        collate_fn=lambda x: x,
        num_workers=min(2, training_args.dataloader_num_workers),
        pin_memory=training_args.dataloader_pin_memory,
    )

    trainer = Qwen2VLGRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        script_args=script_args,
        train_dataset=dataset,
        train_dataloader=data_loader,
        eval_dataset=None,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
    else:
        trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

    # Finish swanlab logging if available
    if _swanlab_available:
        try:
            swanlab.finish()
        except Exception:
            pass


def save_args_to_file(
    script_args, training_args, model_args, filepath="saved_args.json"
):
    """
    将所有解析的参数保存到JSON文件

    Args:
        script_args: GRPOScriptArguments 对象
        training_args: GRPOConfig 对象
        model_args: ModelConfig 对象
        filepath: 保存文件的路径
    """
    import json
    import os
    from dataclasses import asdict

    # 创建包含所有参数的字典
    args_dict = {
        "script_args": asdict(script_args),
        "training_args": training_args.to_dict(),
        "model_args": asdict(model_args),
    }

    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # 保存为JSON文件
    with open(filepath, "w") as f:
        json.dump(args_dict, f, indent=4)

    print(f"所有参数已保存到: {filepath}")


if __name__ == "__main__":
    wait_for_debugger_if_rank0()
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # 保存参数到文件
    save_args_to_file(
        script_args,
        training_args,
        model_args,
        filepath=os.path.join(training_args.output_dir, "all_args.json"),
    )

    main(script_args, training_args, model_args)
