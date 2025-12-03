import fcntl  # 导入文件锁模块
import json
import os
import re
import threading
from pathlib import Path as pth
from typing import Dict, List

import cv2
import numpy as np
from caller import Caller
from PIL import Image

from omni_r1.load_datasets import (
    grounding2sam_process_reward as grounding_process_reward,
)
from omni_r1.load_datasets import (
    grounding_prompt,
    refavs_grounding_process_reward,
    refavs_grounding_prompt,
    sam2_gather_rewards,
    segzero_reward,
    time_reward_multi_time,
    video_r1_reward,
)
from sam_api.webutils import request_predict_with_video

caller = None


VISUALIZATION_PATH = pth("./visualizations")
QUESTION_TYPE = {
    "VideoR1": ["multiple choice", "numerical", "OCR", "free-form", "regression"],
    "SegZero": ["image-segmentation"],
    "VOS": ["video-segmentation"],
    "AVS": ["refavs"],
}


def extract_answer(text):
    pattern = r"<answer>\s*(.*?)\s*</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


_file_locks = {}
_file_locks_lock = threading.Lock()


def output_training_log(
    completion: str,
    gt: str,
    reward: float,
    raw_input: dict,
    output_path: pth = None,
    reward_dict: dict = None,
):
    output_path.mkdir(parents=True, exist_ok=True)

    log_file_path = pth(output_path) / "info.txt"

    with _file_locks_lock:
        if str(log_file_path) not in _file_locks:
            _file_locks[str(log_file_path)] = threading.Lock()
        file_lock = _file_locks[str(log_file_path)]

    with file_lock:
        with open(log_file_path, "a", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                f.write(
                    f"------------- {raw_input['problem_type']} Accuracy reward: {reward} -------------\n"
                )

                f.write(f"Problem: {raw_input['problem']}\n")
                f.write(f"Video Path: {raw_input.get('video_path', 'N/A')}\n")
                f.write(f"Content: {completion}\n")
                if reward_dict is not None:
                    f.write(f"Reward Details: {json.dumps(reward_dict, indent=4)}\n")
                if gt:
                    f.write(f"Solution: {gt}\n")
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)


def accuracy_reward(
    completions: List[str],
    kwargs_list: List[Dict],
    ref_generate=None,
    log_dir=None,
    **alphas,
):
    """Calculate accuracy rewards for model-generated completions.

    Computes reward scores based on different question types (video segmentation, RefAVS, VideoR1, SegZero, etc.).
    This function supports batch processing of multiple completions and optional training log recording.

    Args:
        completions (List[str]): List of model-generated completion results, each containing generated content
        kwargs_list (List[Dict]): List of parameter dictionaries corresponding to each completion,
                                 containing question type, solutions, and other information
        ref_generate (callable, optional): Reference generation function for creating grounding prompts. Defaults to None
        log_dir (str or Path, optional): Directory path for log output. Defaults to None
        **alphas: Variable keyword arguments containing various weight coefficients (e.g., alpha_a, alpha_k, alpha_g)

    Returns:
        List[float]: List of reward scores corresponding to each completion result

    Raises:
        Exception: When errors occur during processing of specific question types,
                  catches and prints error messages, returns 0.0 reward
    """

    log_dir = pth(log_dir).resolve() / "train_logs" if log_dir else None

    question_type = kwargs_list[0]["problem_type"]
    if question_type in QUESTION_TYPE["VOS"]:
        return vos_reward(
            completions=completions,
            kwargs_list=kwargs_list,
            ref_generate=ref_generate,
            log_dir=log_dir,
            **alphas,
        )

    elif question_type in QUESTION_TYPE["AVS"]:
        return refavs_reward(
            completions=completions,
            kwargs_list=kwargs_list,
            ref_generate=ref_generate,
            log_dir=log_dir,
        )

    else:
        rewards = []

        # wrap twice to follow the batch feature of ref_generate
        dummy_prompt = [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Hello!"},
                            # {
                            #     "type": "image",
                            #     "image": kwargs_list[0]['image']
                            # }
                        ],
                    }
                ]
            }
        ]
        _, _ = ref_generate([dummy_prompt])

        for i, kwarg in enumerate(kwargs_list):
            question_type = kwarg["problem_type"]
            content = completions[i][0]["content"]
            sol = kwarg.get("solution", None)
            reward_dict = None
            try:
                if question_type in QUESTION_TYPE["VideoR1"]:
                    output_ans = extract_answer(content)
                    gt_ans = extract_answer(sol)
                    reward = video_r1_reward(
                        question_type=question_type,
                        output_ans=output_ans,
                        gt_ans=gt_ans,
                        **kwarg,
                    )
                    if os.getenv("PLOG") != "true":
                        continue
                    ### dumping debug log info
                    if kwarg["data_type"] == "video":
                        output_path = log_dir / f"{kwarg['problem_id']}"
                        output_path.mkdir(parents=True, exist_ok=True)
                    else:
                        image_cv = cv2.cvtColor(
                            np.array(Image.open(kwarg["path"])), cv2.COLOR_RGB2BGR
                        )
                        output_path = log_dir / f"{kwarg['problem_id']}"
                        output_path.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(output_path / "img.jpg"), image_cv)

                elif question_type in QUESTION_TYPE["SegZero"]:
                    # the completion is extracted between <answer> </answer> tags, but gt is directly the content
                    # so we need to extract the gt answer
                    output_ans = extract_answer(content)

                    reward, sol, ret_output_path = segzero_reward(
                        completion=output_ans,
                        gt=sol,
                        resized_hw=kwarg["preprocess_kwargs"]["img_resized_hw"][0],
                        strict=True,
                        **kwarg,
                    )
                    output_path = ret_output_path if ret_output_path else log_dir

                else:
                    print(
                        f"Unexpected question type, either Unsupported or Routine Bug: {question_type}"
                    )
                    reward = 0.0

            except Exception as e:
                print(f"Error in reward_fn for question_type '{question_type}': {e}")
                reward = 0.0

            rewards.append(reward)

            if os.getenv("LOG_MODE") != "true":
                continue

            output_training_log(
                completion=content,
                gt=sol,
                reward=reward,
                raw_input=kwarg,
                output_path=output_path,
                reward_dict=reward_dict,
            )

        return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents
    ]
    return [1.0 if match else 0.0 for match in matches]


def vos_reward(
    completions: List[str],
    kwargs_list: List[Dict],
    ref_generate=None,
    log_dir=None,
    alpha_a=1.0,
    alpha_k=1.0,
    alpha_g=1.0,
    alpha_k_ratio=1.0,
):
    """Calculate video object segmentation (VOS) rewards for model completions.

    This function computes comprehensive rewards for video object segmentation tasks by combining
    temporal consistency, grounding accuracy, and SAM (Segment Anything Model) IoU scores.
    It processes completions in batches, handles grounding prompts, and optionally uses SAM2
    for enhanced segmentation evaluation.

    Args:
        completions (List[str]): List of model-generated completion results for VOS tasks
        kwargs_list (List[Dict]): List of parameter dictionaries containing video frames,
                                 problem descriptions, and other task-specific information
        ref_generate (callable, optional): Reference generation function for processing grounding prompts.
                                          Defaults to None
        log_dir (str or Path, optional): Directory path for storing training logs. Defaults to None
        alpha_a (float, optional): Weight coefficient for average IoU rewards. Defaults to 1.0
        alpha_k (float, optional): Weight coefficient for temporal consistency rewards
                                  (frame + overlap + ratio). Defaults to 1.0
        alpha_g (float, optional): Weight coefficient for SAM IoU rewards. Defaults to 1.0

    Returns:
        List[float]: List of final weighted reward scores combining temporal, grounding,
                    and segmentation metrics for each completion

    Raises:
        ValueError: When delayed reward index validation fails during SAM processing
        AssertionError: When caller_queue and delay_idx lengths don't match

    Note:
        The function implements a sophisticated reward system that:
        - Extracts temporal information from completions using time_reward_multi_time
        - Generates grounding prompts for valid temporal data
        - Processes grounding rewards through external SAM API calls when alpha_g > 0
        - Combines multiple reward components with configurable weights
    """
    rewards = []
    global caller
    if caller is None:
        caller = Caller(request_predict_with_video)

    dummy_prompt = [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello!",
                        },
                        # {
                        #     "type": "image",
                        #     "image": kwargs_list[0]['frames'][0],
                        # }
                    ],
                }
            ]
        }
    ]

    # 收集所有的 gprompts
    gprompts = []
    cumulated_rewards = []
    # dummy_idx = []
    reward_dicts = []
    caller_queue = []

    for i, (completion, args) in enumerate(zip(completions, kwargs_list)):
        content = completion[0]["content"]
        ans = extract_answer(content)

        cumulated_reward, time_reward_dict = time_reward_multi_time(ans, **args)
        reward_dicts.append(time_reward_dict)
        if time_reward_dict is not None:
            # 处理正常的 prompt
            gprompt = grounding_prompt(
                **time_reward_dict["grounding"], **time_reward_dict["kwargs"]
            )
            gprompts.append(gprompt)
            cumulated_rewards.append(cumulated_reward)
        else:
            # 处理 dummy prompt
            # dummy_idx.append(i)
            gprompts.append(dummy_prompt)
            cumulated_rewards.append(cumulated_reward)

    # 一次性将所有的 gprompts 和 dummy_prompts 输入给 self.reff() 进行批量推理

    gcompletions, gdict_list = ref_generate(gprompts)
    delay_idx = []

    # 根据返回的结果处理每个 gprompt 对应的奖励
    for idx, (completion, args, time_reward_dict, cumulated_reward) in enumerate(
        zip(completions, kwargs_list, reward_dicts, cumulated_rewards)
    ):
        if time_reward_dict is None:  # in dummy_idx:
            # 如果是 dummy prompt，直接使用 cumulated_reward
            rewards.append(cumulated_reward)

        if time_reward_dict is not None:
            final_reward_dict = {}
            ret_dict = grounding_process_reward(
                idx=idx,
                raw_problem=args["problem"],
                inputs=gprompts[idx],
                grounding_indices=time_reward_dict["grounding"]["grounding_indices"],
                valid_mask=time_reward_dict["grounding"]["valid_mask"],
                completions=gcompletions[idx],
                arg_dict=gdict_list[idx],
                **time_reward_dict["kwargs"],
            )
            # FIXME iou_reward type be CallerID
            # iou reward either all be CallerID or all be real value
            if isinstance(ret_dict, dict) and "request" in ret_dict.keys():
                if ret_dict["has_valid_bbox"]:
                    caller_queue.append(ret_dict)
                    rewards.append(-1)
                    delay_idx.append(idx)

                    # 删除不必要的键
                    time_reward_dict["grounding"]["valid_mask"] = (
                        time_reward_dict["grounding"]["valid_mask"]
                        .astype(np.uint8)
                        .tolist()
                    )
                    del time_reward_dict["kwargs"]
                    continue
                else:
                    print(
                        "Error: has_valid_bbox is False, setting rewards to original values"
                    )
                    final_reward_dict = {
                        "cumulated_reward": time_reward_dict["cumulated_reward"],
                        "total_frame_reward": time_reward_dict["total_frame_reward"],
                        "total_overlap_reward": time_reward_dict[
                            "total_overlap_reward"
                        ],
                    }

            elif isinstance(ret_dict, dict):
                for k, v in ret_dict.items():
                    time_reward_dict[k] = v
                final_reward_dict = {
                    "cumulated_reward": time_reward_dict["cumulated_reward"],
                    "total_frame_reward": time_reward_dict["total_frame_reward"],
                    "total_overlap_reward": time_reward_dict["total_overlap_reward"],
                    "avg_iou": ret_dict["avg_iou"],
                    "avg_ratio": ret_dict["avg_ratio"],
                    "object_num_mismatch": ret_dict["object_num_mismatch"],
                }

            else:
                iou_reward = ret_dict
                final_reward_dict = {
                    "cumulated_reward": time_reward_dict["cumulated_reward"],
                    "total_frame_reward": time_reward_dict["total_frame_reward"],
                    "total_overlap_reward": time_reward_dict["total_overlap_reward"],
                    "avg_iou": iou_reward,
                }
                time_reward_dict["iou_reward"] = iou_reward

            # 删除不必要的键
            time_reward_dict["grounding"]["valid_mask"] = (
                time_reward_dict["grounding"]["valid_mask"].astype(np.uint8).tolist()
            )
            del time_reward_dict["kwargs"]

            # rewards.append(time_reward_dict["total_reward"])
            rewards.append(final_reward_dict)

    # 处理 caller_queue
    assert len(caller_queue) == len(delay_idx), (
        f"caller_queue and delay_idx length mismatch: {len(caller_queue)} vs {len(delay_idx)}"
    )
    if len(caller_queue) > 0 and alpha_g > (0.0 + 1e-6):
        responses = caller(caller_queue)
        if responses is None:
            print("Error: responses is None, setting rewards to original values")
            for i in delay_idx:
                rewards[i] = {
                    "cumulated_reward": reward_dicts[i]["cumulated_reward"],
                    "total_frame_reward": reward_dicts[i]["total_frame_reward"],
                    "total_overlap_reward": reward_dicts[i]["total_overlap_reward"],
                }
        else:
            delayed_rewards = sam2_gather_rewards(responses)
            for i, delayed_i in enumerate(delay_idx):
                if rewards[delayed_i] != -1:
                    raise ValueError(
                        f"Delayed Reward at index {i} is not -1, but {rewards[delayed_i]}"
                    )

                rewards[delayed_i] = {
                    "cumulated_reward": reward_dicts[delayed_i]["cumulated_reward"],
                    "total_frame_reward": reward_dicts[delayed_i]["total_frame_reward"],
                    "total_overlap_reward": reward_dicts[delayed_i][
                        "total_overlap_reward"
                    ],
                    "avg_iou": responses[i]["avg_iou"],
                    "avg_ratio": responses[i]["avg_ratio"],
                    "object_num_mismatch": responses[i]["object_num_mismatch"],
                    "sam_iou": delayed_rewards[i],
                }

                reward_dicts[delayed_i]["SAM_IOU"] = delayed_rewards[i]

    # 处理完毕，清空 caller_queue
    elif len(caller_queue) > 0:
        for i, delayed_i in enumerate(delay_idx):
            if rewards[delayed_i] != -1:
                raise ValueError(
                    f"Delayed Reward at index {i} is not -1, but {rewards[delayed_i]}"
                )
            rewards[delayed_i] = {
                "cumulated_reward": reward_dicts[delayed_i]["cumulated_reward"],
                "total_frame_reward": reward_dicts[delayed_i]["total_frame_reward"],
                "total_overlap_reward": reward_dicts[delayed_i]["total_overlap_reward"],
            }

    caller_queue = []
    final_rewards = handle_rewards(
        rewards=rewards,
        alpha_a=alpha_a,
        alpha_k=alpha_k,
        alpha_k_ratio=alpha_k_ratio,
        alpha_g=alpha_g,
        completions=completions,
        kwargs_list=kwargs_list,
        output_path=log_dir,
    )

    return final_rewards


def handle_rewards(
    rewards,
    alpha_a,
    alpha_k,
    alpha_g,
    alpha_k_ratio,
    completions,
    kwargs_list,
    output_path,
):
    final_rewards = []

    for i, reward in enumerate(rewards):
        if isinstance(reward, dict):
            cumulated_reward = reward.get("cumulated_reward", 0.0)
            total_frame_reward = reward.get("total_frame_reward", 0.0)
            total_overlap_reward = reward.get("total_overlap_reward", 0.0)
            avg_iou = reward.get("avg_iou", 0.0)
            avg_ratio = reward.get("avg_ratio", 0.0)
            object_num_mismatch = reward.get("object_num_mismatch", 0.0)
            sam_iou = reward.get("sam_iou", 0.0)

            # this is the naive reward calculation
            final_reward = (
                cumulated_reward
                + alpha_k * (total_frame_reward + total_overlap_reward)
                + alpha_k_ratio * avg_ratio
                + alpha_a * avg_iou
                + object_num_mismatch
                + alpha_g * sam_iou
            )
            #
            is_above = (
                alpha_k * (total_frame_reward + total_overlap_reward)
                + alpha_k_ratio * avg_ratio
                + alpha_a * avg_iou
                + object_num_mismatch
                + alpha_g * sam_iou
            ) > 0.0
            if not is_above and os.getenv("CLIP_REWARD") == "1":
                print("Clipping rewards due to large penelty...")
                final_reward = cumulated_reward

        else:
            final_reward = reward

        final_rewards.append(final_reward)

        if os.getenv("LOG_MODE") != "true":
            continue

        output_training_log(
            completions[i][0]["content"],
            None,
            final_reward,
            kwargs_list[i],
            output_path=output_path,
            reward_dict=reward if isinstance(reward, dict) else None,
        )
    return final_rewards


def refavs_reward(
    completions: List[str], kwargs_list: List[Dict], ref_generate=None, log_dir=None
):
    rewards = []

    dummy_prompt = [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello!",
                        }
                        # {
                        #     "type": "image",
                        #     "image": kwargs_list[0]['frames'][0],
                        # }
                    ],
                }
            ]
        }
    ]

    # 收集所有的 gprompts
    gprompts = []

    # dummy_idx = []

    valid_list = []

    for i, (completion, args) in enumerate(zip(completions, kwargs_list)):
        content = completion[0]["content"]
        ans = extract_answer(content)

        if len(ans) > 0:
            # 处理正常的 prompt
            gprompt = refavs_grounding_prompt(ans, **args)
            gprompts.append(gprompt)
            valid_list.append(True)

        else:
            # 处理 dummy prompt
            # dummy_idx.append(i)
            gprompts.append(dummy_prompt)
            valid_list.append(False)

    # 一次性将所有的 gprompts 和 dummy_prompts 输入给 self.reff() 进行批量推理

    gcompletions, gdict_list = ref_generate(gprompts)

    # 根据返回的结果处理每个 gprompt 对应的奖励
    for idx, (completion, args, is_valid) in enumerate(
        zip(completions, kwargs_list, valid_list)
    ):
        if not is_valid:  # in dummy_idx:
            rewards.append(0.0)

        else:
            iou_reward = refavs_grounding_process_reward(
                idx=idx,
                raw_problem=args["problem"],
                input=gprompts[idx][0],
                completion=gcompletions[idx][0],
                grounding_arg=gdict_list[idx][0],
                raw_args=args,
            )
            rewards.append(iou_reward)

        if os.getenv("LOG_MODE") != "true":
            continue
        # 打印训练日志
        output_training_log(
            completion[0]["content"],
            gt=None,
            reward=0.0 if not is_valid else iou_reward,
            raw_input=args,
            output_path=log_dir,
            reward_dict=None,
        )
    return rewards
