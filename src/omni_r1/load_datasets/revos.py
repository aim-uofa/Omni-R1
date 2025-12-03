import copy
import json
import logging
import os
import random
import re
import time
from pathlib import Path as pth
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import pycocotools.mask as maskUtils
import torch
from datasets import Dataset
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt

from omni_r1.eval_utils import db_eval_iou

# QUESTION_TEMPLATE = (
#     'Given a video containing {frames} frames, please find the objects described as "{Question}" in these video frames with bbox and points.'
#     "Compare all frames and identify each target object consistently."
#     "Output your thinking process in <think> </think> tags and final grounding results in <answer> </answer> tags."
#     "Output the one bbox of each interested object for each frame using JSON format."
#     "The grounding result should be in the format of json. 0, 1,..., frame_num-1 indicate frame idx."
#     "If the object is not found in the frame, please output a dummy bbox as [-1, -1, -1, -1]"
#     "e.g. if only one object is found in a two frame video, you should response as: {AnswerOne}."  # \n If two or more objects found: {AnswerMore}."
# )

VIDEO_TIMESTAMP_TEMPLATE = (
    "Given a [frames] seconds video and a reference instruction: [ref_prompt] that may involve temporal behavior, "
    "identify the exact object(s) [ref_prompt] that matches the description in the video. "
    "Select the most relevant 1 seconds duration or less that contains the referred object and simple to recognize it. "
    "Then, simplify the identified object into a short and clear visual grounding description that can be used for single-image understanding. "
    "Avoid temporal phrases and comparison phrases like 'walking', 'moving', 'approaching', 'bigger' or 'smaller', and instead describe visible visual cues like clothing, pose, position, or grouping. "
    "Explain your reasoning in <think>...</think> and output the final result in <answer>...</answer>. "
    "Note that you do not locate from beginning, but the most relevant durations of highly representative moments. "
    "Your final answer should be a JSON object in the following format:\n"
    "<think> think process here </think>\n"
    "<answer>\n"
    "{\n"
    '  "start_time": "00:[start]",'
    '  "end_time": "00:[end]",'
    '  "description": "direct description of referred object"\n'
    "}\n"
    "//another duration in json format if necessary\n"
    "</answer>\n"
)


VIDEO_TIMESTAMP_FORCE_TEMPLATE = (
    "Given a [frames] seconds video and a reference instruction: [ref_prompt] that may involve temporal behavior, "
    "identify the exact object(s) [ref_prompt] in the video that matches the description. "
    "Select about 4 most relevant moments that contain the referred object(s) with the best view. "
    "Then, simplify the identified object into a short and clear visual grounding description that can be used for single-image reference at each moment. "
    "Avoid temporal phrases and comparison phrases like 'walking', 'moving', 'approaching', 'bigger' or 'smaller', but instead describe visible visual cues like clothing, pose, position, or grouping. "
    "Try to select moments that are **temporally well-distributed across the video**, rather than clustered in the same part of the timeline. "
    "Avoid selecting multiple timestamps that are adjacent or overlapping; instead, prefer clearly distinct moments that each offer unique visual information. "
    "It's better to choose the most relevant and highly representative moments **spanning the entire video**, rather than picking all from the beginning. "
    "Explain your reasoning in <think>...</think> and output the final result in <answer>...</answer>. "
    "Your final answer should be a JSON object in the following format, which is an answer containing 4 moments:\n"
    "<think> your analysis about the video and reference instruction </think>\n"
    "<answer>\n"
    "{\n"
    '  "start_time": "00:[start]",\n'
    '  "end_time": "00:[end]",\n'
    '  "description": "direct description of referred object(s) at this moment"\n'
    "}\n"
    "{\n"
    '  "start_time": "00:[start1]",\n'
    '  "end_time": "00:[end1]",\n'
    '  "description": "direct description of referred object(s) at this moment"\n'
    "}\n"
    "{\n"
    '  "start_time": "00:[start2]",\n'
    '  "end_time": "00:[end2]",\n'
    '  "description": "direct description of referred object(s) at this moment"\n'
    "}\n"
    "{\n"
    '  "start_time": "00:[start3]",\n'
    '  "end_time": "00:[end3]",\n'
    '  "description": "direct description of referred object(s) at this moment"\n'
    "}\n"
    "</answer>\n"
)

VIDEO_TIMESTAMP_NO_HINT_TEMPLATE = (
    "Given a [frames] seconds video and a reference instruction: [ref_prompt] that may involve temporal behavior, "
    "identify the exact object(s) [ref_prompt] in the video that matches the description. "
    "Select about 1-4 most relevant moments that contain the referred object(s) with the best view. "
    "Then, simplify the identified object into a short and clear visual grounding description that can be used for single-image reference at each moment. "
    "Avoid temporal phrases and comparison phrases like 'walking', 'moving', 'approaching', 'bigger' or 'smaller', but instead describe visible visual cues like clothing, pose, position, or grouping. "
    "Try to select moments that are **temporally well-distributed across the video**, rather than clustered in the same part of the timeline. "
    "Avoid selecting multiple timestamps that are adjacent or overlapping; instead, prefer clearly distinct moments that each offer unique visual information. "
    "It's better to choose the most relevant and highly representative moments **spanning the entire video**, rather than picking all from the beginning. "
    "Explain your reasoning in <think>...</think> and output the final result in <answer>...</answer> in json format with 'mm:ss.ff' format for time depiction."
)


JSON_EXAMPLE = (
    '{ "0": { "bbox_2d": [10, 40, 100, 150]},"1": {"bbox_2d": [60, 90, 180, 230]}}'
)

ANSWER_ONE = "<think> think process here </think> <answer> object_1: {json_example}, object_2(if there exists): [the same json format] </answer>"


# ORIGINAL_GROUNDING_TEMPLATE = (
#     "Please find '{Question}' with bbox."
#     "Compare the difference between objects and find the most closely matched one."
#     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
#     "Output the bbox of the most interested object in JSON format."
#     "i.e., <think> thinking process here </think>"
#     "<answer>{Answer}</answer>"
# )

# origin_grounding_answer = '{ "bbox_2d" : [x1,y1,x2,y2] }'

# NOTHINKING_GROUNDING_TEMPLATE = (
#     "Please find '{Question}' with bbox."
#     "Compare the difference between objects and find the most closely matched one."
#     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
#     "Output the bbox of the most interested object in JSON format."
#     "i.e., <think> thinking process here </think>"
#     "<answer>{Answer}</answer>"
# )


class VideoReVOSDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        dataset_path: Union[str, pth],
        select_number=1,
        min_frames=8,
        max_frames=24,
        use_prompt_forcing=True,  # 新增参数：是否使用时间戳提示
        image_folder: str = None,
        expression_file: str = None,
        mask_file: str = None,
        *args,
        **kwargs,
    ):
        dataset_folder = pth(dataset_path).resolve()
        self.dataset_folder = dataset_folder

        if not dataset_folder.exists():
            raise ValueError(f"Dataset folder {dataset_folder} does not exist.")

        self.image_folder = (
            dataset_folder if image_folder is None else pth(image_folder).resolve()
        )
        self.expression_file = (
            (dataset_folder / "meta_expressions_train_.json")
            if expression_file is None
            else pth(expression_file).resolve()
        )
        self.mask_file = (
            (dataset_folder / "mask_dict.json")
            if mask_file is None
            else pth(mask_file).resolve()
        )

        self.select_number = select_number
        # print(
        #     f"################# Object Number up to: {self.select_number} ################# "
        # )

        self.min_frames = min_frames
        self.max_frames = max_frames

        # print(
        #     f"################# min_frames: {self.min_frames}, max_frames: {self.max_frames} ################# "
        # )

        self.use_prompt_forcing = use_prompt_forcing

        self.lazy = False
        self.iter_step = 0

        self.load_dataset()

    def get_supplementary_dataset(self, expression_file_path):
        return VideoReVOSDataset(
            dataset_path=self.dataset_folder,
            image_folder=self.image_folder,
            expression_file=expression_file_path,
            use_prompt_forcing=self.use_prompt_forcing,
            mask_file=self.mask_file,
            max_frames=self.max_frames,
            min_frames=self.min_frames,
            select_number=self.select_number,
        )

    def load_dataset(self):
        """只加载元数据，不处理样本"""
        # 加载表达式数据
        with open(self.expression_file, "r") as f:
            expression_datas = json.load(f)["videos"]

        # 加载掩码数据
        with open(self.mask_file, "rb") as f:
            self.mask_dict = json.load(f)

        metas = []
        anno_count = 0  # anno_id
        vid2metaid = {}  # video_name -> meta_id

        # 处理每个视频的元数据
        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]
            vid_frames = sorted(vid_express_data["frames"])
            vid_len = len(vid_frames)
            exp_id_list = sorted(list(vid_express_data["expressions"].keys()))

            for exp_id in exp_id_list:
                exp_dict = vid_express_data["expressions"][exp_id]
                meta = {}
                meta["video"] = vid_name
                meta["exp"] = exp_dict["exp"]
                meta["mask_anno_id"] = exp_dict["anno_id"]

                if "obj_id" in exp_dict.keys():
                    meta["obj_id"] = exp_dict["obj_id"]
                else:
                    meta["obj_id"] = [
                        0,
                    ]  # 只有一个对象

                meta["anno_id"] = [
                    str(anno_count),
                ]
                anno_count += 1
                meta["frames"] = vid_frames
                meta["exp_id"] = exp_id
                meta["length"] = vid_len

                metas.append(meta)
                if vid_name not in vid2metaid:
                    vid2metaid[vid_name] = []
                vid2metaid[vid_name].append(len(metas) - 1)

        self.vid2metaid = vid2metaid
        self.videos = list(self.vid2metaid.keys())
        self.metas = metas

        logging.info(
            f"Loaded metadata for {len(self.videos)} videos and {len(self.metas)} expressions"
        )

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        """return batched data for video segmentation dataset

        Args:
            idx (list(int)): _description_

        Returns:
            dict: {
                'data_type': video
                'problem_type': video-segmentation,
                'problem_id': video_meta['anno_id'], reassigned in creating process, not the original anno_id in meta_expression
                'frames': list of frame image path,
                'masks': video_masks object-wise as np.array (obj_num, frames, H, W)
                'obj_number': number of objects in each video, list of int, [obj1_number, obj2_number, ...]
                'bboxs': same of structure of 'masks', shaped as (obj_num, frames, 4)
                'points': same of structure of 'masks', shaped as (obj_num, frames, 4)
                'expression': expression str,
                'prompt': prompt dict, refer to _create_prompt()

            }
        """

        idx = idx % len(self.videos)  # 确保索引在范围内

        video_name = self.videos[idx]

        # 获取视频对应的expression meta index list
        video_objects = self.vid2metaid[video_name]

        video_meta = []
        # 获取选定对象的元数据, 存在一些meta项，object_id为空，跳过
        for i in video_objects:
            if len(self.metas[i]["obj_id"]) == 0:
                continue
            video_meta.append(self.metas[i])

        item = self._process_single_item(video_meta)

        self.iter_step = self.iter_step + 1
        return item

    def _process_single_item(self, video_meta):
        """处理单个视频数据项"""
        # random pick one expression
        selected_meta = random.choice(video_meta)

        # 采用宽松的视频帧处理，尽量多些帧数
        selected_frame_indices = list(range(len(selected_meta["frames"])))
        if len(selected_frame_indices) > self.min_frames:
            f_num = random.randint(self.min_frames, self.max_frames)
            selected_frame_indices = np.linspace(
                0, len(selected_meta["frames"]) - 1, f_num, dtype=int
            ).tolist()

        # 获取帧路径
        frames = []

        for idx in selected_frame_indices:
            frame_id = selected_meta["frames"][idx]
            frame_path = os.path.join(
                self.image_folder, selected_meta["video"], frame_id + ".jpg"
            )
            frames.append(frame_path)

        with Image.open(frames[0]) as img:
            ori_wh = img.size
        ori_hw = (ori_wh[1], ori_wh[0])

        # 处理掩码 - 仅使用选定表达式的掩码
        video_masks = []
        video_bboxs = []
        video_points = []

        anno_ids = selected_meta["mask_anno_id"]

        obj_upper = (
            self.select_number if len(anno_ids) > self.select_number else len(anno_ids)
        )
        pick_idx = np.random.choice(len(anno_ids), obj_upper, replace=False).tolist()

        # full_masks_ = []
        for idx in pick_idx:
            anno_id = anno_ids[idx]
            anno_id = str(anno_id)
            frames_masks = self.mask_dict[anno_id]
            frames_masks_ = []
            for frame_idx in selected_frame_indices:
                # frame_idx_str = str(frame_idx)
                # if frame_idx_str in frames_masks:
                #     frames_masks_.append(copy.deepcopy(frames_masks[frame_idx_str]))
                # else:
                #     frames_masks_.append(None)  # 处理缺失帧
                frames_masks_.append(copy.deepcopy(frames_masks[frame_idx]))
            # full_masks_.append(frames_masks)

            masks = self.decode_mask(frames_masks_, ori_hw)
            # full_masks = self.decode_mask(full_masks_, ori_hw)
            # full_video_masks.append(full_masks)
            video_masks.append(masks)
            bboxs = generate_bbox(masks)
            video_bboxs.append(bboxs)
            points = generate_points(masks)
            video_points.append(points)

        # video_masks: list [masks_1, masks_2, ...]
        # masks_i: np.array [frame_1, frame_2, ...] if certain frame is null, it's zeros in masks_i array.

        try:
            video_masks = np.stack(video_masks, axis=0)  # [obj_num, frame_num, H, W]
            # full_video_masks = np.stack(full_video_masks, axis=0)  # [obj_num, frame_num, H, W]
        except Exception:
            print("ERROR, Masks' size mismatching. Fixing...")
            # for m in video_masks:
            #     if m.shape[1] != ori_hw[0] or m.shape[2] != ori_hw[1]:
            #         size = (m.shape[1], m.shape[2])

            for i, m in enumerate(video_masks):
                if m.shape[1] != ori_hw[0] or m.shape[2] != ori_hw[1]:
                    video_masks[i] = (
                        torch.nn.functional.interpolate(
                            torch.from_numpy(m[None, :, :, :]).float(),
                            size=ori_hw,
                            mode="nearest",
                        )
                        .int()
                        .cpu()
                        .squeeze(0)
                        .numpy()
                        .astype(np.uint8)
                    )
            video_masks = np.stack(video_masks, axis=0)  # [obj_num, frame_num, H, W]
        video_bboxs = np.stack(video_bboxs, axis=0)  # [obj_num, frame_num, 4]
        video_points = np.stack(video_points, axis=0)  # [obj_num, frame_num, 4]

        # 生成问题
        expression: str = selected_meta["exp"]
        # question: str = random.choice(SEG_QUESTIONS).format(class_name=expression.lower())
        prompt = self._create_prompt(frames, expression)

        # 准备返回数据
        result = {
            "data_type": "video",
            "problem_type": "video-segmentation",
            "problem_id": selected_meta["anno_id"][0],
            "frames": frames,
            "video_path": str(os.path.join(self.image_folder, selected_meta["video"])),
            "obj_number": obj_upper,  # len(anno_ids),
            "masks": video_masks,
            #            "full_masks": full_video_masks,
            "bboxs": video_bboxs,
            "points": video_points,
            "expression": expression,
            "problem": expression,
            "prompt": prompt,
        }

        return result

    def _create_prompt(self, frames, expression):
        """创建提示格式"""
        # 创建内容列表，先添加所有视频帧
        content = []

        # TODO: See qwen-vl-utils/src/visual_process.py:330
        content.append(
            {
                "type": "video",
                "video": [Image.open(frame_path) for frame_path in frames]
                if not self.lazy
                else frames,
            }
        )

        # 添加文本问题
        if self.use_prompt_forcing:
            # FIXME: temporarily disable timestamp prompt
            start = random.randint(0, len(frames) - 2)
            end = (min(start + random.randint(1, 4), len(frames) - 1)) * 0.5
            start = start * 0.5

            start1 = random.randint(0, len(frames) - 2)
            end1 = (min(start1 + random.randint(1, 4), len(frames) - 1)) * 0.5
            start1 = start1 * 0.5

            start2 = random.randint(0, len(frames) - 2)
            end2 = (min(start2 + random.randint(1, 4), len(frames) - 1)) * 0.5
            start2 = start2 * 0.5

            start3 = random.randint(0, len(frames) - 2)
            end3 = (min(start3 + random.randint(1, 4), len(frames) - 1)) * 0.5
            start3 = start3 * 0.5

            end = start
            end1 = start1
            end2 = start2
            end3 = start3

            content.append(
                {
                    "type": "text",
                    "text": VIDEO_TIMESTAMP_FORCE_TEMPLATE.replace(
                        "[ref_prompt]", "[" + expression + "]"
                    )
                    .replace("[frames]", str(len(frames) / 2))
                    .replace("[start]", f"{start:05.2f}")
                    .replace("[end]", f"{end:05.2f}")
                    .replace("[start1]", f"{start1:05.2f}")
                    .replace("[end1]", f"{end1:05.2f}")
                    .replace("[start2]", f"{start2:05.2f}")
                    .replace("[end2]", f"{end2:05.2f}")
                    .replace("[start3]", f"{start3:05.2f}")
                    .replace("[end3]", f"{end3:05.2f}"),
                }
            )

        else:
            example1 = (
                "for example,\n"
                "<think> your analysis about the video and reference instruction, and you pick one timestamp </think>\n"
                "<answer>\n"
                "{\n"
                '"start_time": "mm:ss.ff",\n'
                '"end_time": "mm:ss.ff",\n'
                '"description": "direct description of referred object(s) at this moment"\n'
                "}\n"
                "</answer>\n"
            )
            example2 = (
                "for example,\n"
                "<think> your analysis about the video and reference instruction, and you pick two timestamps </think>\n"
                "<answer>\n"
                "{\n"
                '"start_time": "mm:ss.ff",\n'
                '"end_time": "mm:ss.ff",\n'
                '"description": "direct description of referred object(s) at this moment"\n'
                "},\n"
                "{\n"
                '"start_time": "mm:ss.ff",\n'
                '"end_time": "mm:ss.ff",\n'
                '"description": "direct description of referred object(s) at this moment"\n'
                "}\n"
                "</answer>\n"
            )
            text = VIDEO_TIMESTAMP_NO_HINT_TEMPLATE.replace(
                "[ref_prompt]", "[" + expression + "]"
            ).replace("[frames]", str(len(frames) / 2))

            if self.iter_step < 50:
                example = random.choice([example1, example2])
                text = text + example

            content.append(
                {
                    "type": "text",
                    "text": text,
                }
            )

        # 创建提示
        prompt = [{"role": "user", "content": content}]

        return prompt

    def decode_mask(self, video_masks, image_size):
        """解码掩码"""
        try:
            masks: np.array = (
                maskUtils.decode(video_masks).permute(2, 0, 1).astype(np.uint8)
            )
        except Exception:
            masks = []
            for mask_frame in video_masks:
                # 处理空对象
                if mask_frame is None:
                    if len(masks) != 0:
                        mask = masks[0] * 0
                    else:
                        mask = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
                else:
                    mask = maskUtils.decode(mask_frame)
                    if len(masks) != 0 and mask.shape != masks[-1].shape:
                        if masks[-1].sum() != 0:
                            print("Warning: mask shape mismatch, forcing to resize")
                        for i, m in enumerate(masks):
                            if m.sum() == 0:
                                masks[i] = np.zeros(mask.shape, dtype=np.uint8)
                            else:
                                mask[i] = np.array(
                                    Image.fromarray(m).resize(mask.shape, Image.NEAREST)
                                ).astype(np.uint8)
                masks.append(mask)

            masks = np.stack(masks, axis=0)

        return masks


def top2_inscribed_circle_centers(mask, min_distance=10):
    """
    Finds the centers of the two largest inscribed circles within a binary mask.

    Args:
        mask (np.ndarray): 2D binary array (H, W), dtype=bool or int
        min_distance (int): Minimum pixel distance between the two points (optional)

    Returns:
        List of two (y, x) tuples
    """
    assert mask.ndim == 2, "Mask must be 2D"
    # return dummy bbox if mask is empty
    if not mask.any():
        return np.array([-1, -1, -1, -1])

    # Compute Euclidean distance transform
    dist_map = distance_transform_edt(mask)

    # First center: global maximum
    first_yx = np.unravel_index(np.argmax(dist_map), dist_map.shape)

    # Mask out a circular region around the first point to find a second distant peak
    yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
    mask_out = ((yy - first_yx[0]) ** 2 + (xx - first_yx[1]) ** 2) <= min_distance**2
    # dist_map_masked = dist_map.copy()
    # dist_map_masked[mask_out] = 0

    mask_masked = mask.copy()
    mask_masked[mask_out] = 0
    dist_map_masked = distance_transform_edt(mask_masked)
    # Second center: next local maximum
    second_yx = np.unravel_index(np.argmax(dist_map_masked), dist_map.shape)

    # Sanity check
    if dist_map[second_yx] == 0:
        # Randomly pick one more point inside the mask
        valid_points = np.argwhere(mask)
        random_point = valid_points[np.random.choice(len(valid_points))]
        return np.array([first_yx[1], first_yx[0], random_point[1], random_point[0]])
        # return [first_yx]  # Only one valid region

    return np.array([first_yx[1], first_yx[0], second_yx[1], second_yx[0]])


def generate_bbox(masks: np.array):
    """生成边界框"""
    if masks.ndim != 3:
        raise ValueError("Masks should be 3D tensor")

    bboxes = []
    for mask in masks:
        if mask.sum() == 0:
            # dummy bbox for empty mask
            bboxes.append([-1, -1, -1, -1])
            continue
        pos = np.where(mask > 0)
        x1, y1, x2, y2 = pos[1].min(), pos[0].min(), pos[1].max(), pos[0].max()
        bboxes.append([x1, y1, x2, y2])
    return np.array(bboxes)


def generate_points(masks: np.array):
    assert masks.ndim == 3, "Masks should be 3D np.array"
    p_list = []
    for mask in masks:
        if mask.sum() == 0:
            p_list.append(np.array([-1, -1, -1, -1]))
            continue
        points = top2_inscribed_circle_centers(mask, min_distance=10)
        p_list.append(points)

    return np.stack(p_list, axis=0)


def make_prompt(frames: List[str], descriptions: List[str]) -> dict:
    """生成用于Grounding的提示词

    Args:
        frames (List[str]): 输入的帧路径列表
        description (str): 描述文本

    Returns:
        dict: 包含提示词的字典
    """
    # img_num = len(frames)
    VISIONREASONER_TEMPLATE = (
        'Please find "{Question}" with bboxs and points.'
        "Compare the difference between object(s) and find the most closely matched object(s)."
        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        "Output the bbox(es) and point(s) inside the interested object(s) in JSON format."
        "i.e., <think> thinking process here </think>"
        "<answer>{Answer}</answer>"
    )

    OMNI_SEGZERO_QUESTION_TEMPLATE = (
    "Please find '{Question}' with bbox and points."
    "Compare the difference between objects and find the most closely matched one."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format."
    "i.e., <think> thinking process here </think>"
    "<answer>{Answer}</answer>"
)

    inputs = []

    for i, (frame, description) in enumerate(zip(frames, descriptions)):
        if os.getenv('USE_GROUNDING_PROMPT_VISIONRESONER') == '1':
            text = VISIONREASONER_TEMPLATE.format(
                Question=description.lower().strip("."),
                Answer='[{"bbox_2d": [10,100,200,210], "point_2d": [30,110]}, {"bbox_2d": [225,296,706,786], "point_2d": [302,410]}]',
            )
        elif os.getenv('USE_GROUNDING_PROMPT_OMNI_SEGZERO') == '1':
            text = OMNI_SEGZERO_QUESTION_TEMPLATE.format(
                Question=description.lower().strip("."),
                Answer='{"bbox_2d": [x1, y1, x2, y2], "points_1": [x3, y3], "points_2": [x4, y4]}',
            )
            
        else:
            text = "Given the expression: '{Question}', ground all the referred objects with json bboxs".format(
                Question=description.lower().strip()
            )
        
        input = {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": frame,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]
        }
        inputs.append(input)

    return inputs


def write_log(
    nidx: int,
    raw_problem: str,
    inputs: list,
    completions: list,
    arg_dict: list,
    valid_mask: np.array,
    ious: np.array,
    pred_bboxs: np.array,
    scaled_bboxs: np.array,
):
    """将输入和输出写入日志文件，支持多目标框标注"""

    # Get valid indices for frames
    idx = np.array(np.nonzero(valid_mask.squeeze())).tolist()[0]

    tstr = time.strftime("%Y%m%d-%H%M%S")

    for i, (input, completion, data_info) in enumerate(
        zip(inputs, completions, arg_dict)
    ):
        if i not in idx:
            continue

        img_resized_hw = data_info.get("img_resized_hw", None)[0]
        img_ori_hw = data_info.get("img_ori_hw", None)[0]
        if img_resized_hw is None or img_ori_hw is None:
            raise ValueError("img_resized_hw or img_ori_hw is None")

        # Assume obj_num is > 1, iterating over objects
        img = Image.open(input["prompt"][0]["content"][0]["image"]).resize(
            (img_resized_hw[1], img_resized_hw[0]), Image.Resampling.BICUBIC
        )
        draw = ImageDraw.Draw(img)

        # Define colors
        bbox_color = (255, 0, 0)  # Red for predicted bounding boxes
        gt_color = (0, 255, 0)  # Green for ground truth
        text_color = (255, 255, 255)  # White for text

        # Iterate over objects (obj_num > 1)
        for obj_id in range(pred_bboxs.shape[0]):  # Loop over objects
            pred_bbox = pred_bboxs[obj_id, i]  # Predicted bbox for the object
            scaled_bbox = scaled_bboxs[obj_id, i]  # Ground truth bbox for the object

            # Draw predicted bbox
            if not (pred_bbox == -1).all():
                x1, y1, x2, y2 = tuple(pred_bbox.squeeze().tolist())
                draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=2)
                # Optionally, add the object ID as text on the bbox
                draw.text((x1, y1), str(obj_id), fill=text_color)

            # Draw ground truth bbox
            if not (scaled_bbox == -1).all():
                x1, y1, x2, y2 = tuple(scaled_bbox.squeeze().tolist())
                draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=2)
                # Optionally, add the object ID as text on the bbox
                draw.text((x1, y1 + 10), str(obj_id), fill=text_color)

        # Save the annotated image for debugging
        output_dir = pth(os.getenv("LOG_PATH")).resolve() / (
            tstr + "ROLL_" + str(nidx) + "_R" + str(torch.distributed.get_rank())
        )
        os.makedirs(output_dir, exist_ok=True)

        # Save image with IOU value in the filename
        img.save(
            os.path.join(
                output_dir, f"frame_{i}_iou_{np.mean(ious[:, i], axis=0):.2f}.jpg"
            )
        )

        # Write additional information to a text file
        with open((output_dir / "info.txt"), "a", encoding="utf-8") as f:
            f.write(f"--------------------Frame {i}:--------------------\n")
            f.write(f"Raw Problem:\n {raw_problem}\n")
            f.write(f"File Path: {input['prompt'][0]['content'][0]['image']}\n")
            f.write(f"Input:\n {input}\n")
            f.write(f"Completion:\n {completion}\n")
            f.write(f"Predicted IOU: {np.mean(ious[:, i], axis=0):.2f}\n")


def grounding_prompt(grounding_indices: list, descriptions: list, **kwargs):
    frame_paths = kwargs.get("frames", None)
    assert frame_paths is not None, "frame paths should not be None"
    assert len(grounding_indices) == len(descriptions), (
        "grounding_indices and descriptions should have the same length"
    )
    selected_frames = [frame_paths[i] for i in grounding_indices]
    inputs = make_prompt(selected_frames, descriptions)
    return inputs


def iou_framewise_object_matching(
    gt_bboxs: np.array, pred_bboxs: np.array
) -> Tuple[np.array, np.array]:
    """
    为GT masks和预测边界框找到最佳的对象级别匹配，逐帧使用匈牙利算法计算IoU和L1距离。

    Args:
        gt_bboxs: GT bbox列表，形状为 [obj_num, frame_num, 4]
        pred_bboxs: 预测的边界框，形状为 [obj_num, frame_num, 4]

    Returns:
        np.array: 匹配后的IoU矩阵，形状为 [obj_num, frame_num]，gt_bboxs[i] 与其最佳匹配的 pred_bbox 在第 j 帧的 IoU
        np.array: 匹配后的L1距离矩阵，形状为 [obj_num, frame_num]，gt_bboxs[i] 与其最佳匹配的 pred_bbox 在第 j 帧的 L1 距离
        np.array: permute_matrix： 匹配结果, [frame_num, obj_num]
        np.array: permuted_bboxs = 匹配之后，对齐gt的bboxs结果
    """
    assert pred_bboxs.ndim == 3, (
        "pred_bboxs should be 3D np.array of (obj_num, frame_num, 4)"
    )
    assert gt_bboxs.shape[0] == pred_bboxs.shape[0], (
        "gt_bboxs and pred_bboxs should have the same number of objects"
    )
    assert gt_bboxs.ndim == 3, (
        "gt_bboxs should be 3D np.array of (obj_num, frame_num, 4)"
    )
    assert gt_bboxs.shape == pred_bboxs.shape, (
        "gt_bboxs and pred_bboxs should have the same shape"
    )

    obj_num, frame_num = gt_bboxs.shape[0], gt_bboxs.shape[1]

    # 初始化存储匹配后结果的矩阵
    matched_iou_matrix = np.zeros((obj_num, frame_num), dtype=np.float32)
    matched_l1_matrix = np.full(
        (obj_num, frame_num), -1.0, dtype=np.float32
    )  # 使用-1表示无匹配或无效
    permuted_bboxs = copy.deepcopy(pred_bboxs)
    permutation = []

    try:
        from scipy.optimize import linear_sum_assignment

        use_scipy = True
    except ImportError:
        use_scipy = False
        print("scipy not available, using greedy matching for frame-wise assignment")

    # 逐帧处理
    for f_idx in range(frame_num):
        gt_boxes_frame = gt_bboxs[:, f_idx, :]  # [obj_num, 4]
        pred_boxes_frame = pred_bboxs[:, f_idx, :]  # [obj_num, 4]

        # 找出当前帧有效的 GT 和 Pred 对象索引 (非 dummy box)
        # valid_gt_indices = np.where(~np.all(gt_boxes_frame == -1, axis=-1))[0]
        # valid_pred_indices = np.where(~np.all(pred_boxes_frame == -1, axis=-1))[0]
        valid_gt_indices = np.arange(obj_num)
        valid_pred_indices = np.arange(obj_num)

        num_valid_gt = len(valid_gt_indices)
        num_valid_pred = len(valid_pred_indices)

        # 如果当前帧没有有效的 GT 或 Pred 对象，则跳过
        if num_valid_gt == 0 or num_valid_pred == 0:
            continue

        # 提取有效的边界框
        valid_gt_boxes = gt_boxes_frame[valid_gt_indices]  # [num_valid_gt, 4]
        valid_pred_boxes = pred_boxes_frame[valid_pred_indices]  # [num_valid_pred, 4]

        # 计算当前帧的 IoU 矩阵和 L1 距离矩阵
        iou_matrix_frame = np.zeros((num_valid_gt, num_valid_pred), dtype=np.float32)
        l1_matrix_frame = np.zeros((num_valid_gt, num_valid_pred), dtype=np.float32)

        for i, gt_box in enumerate(valid_gt_boxes):
            for j, pred_box in enumerate(valid_pred_boxes):
                # 计算 IoU
                inter_x1 = np.maximum(gt_box[0], pred_box[0])
                inter_y1 = np.maximum(gt_box[1], pred_box[1])
                inter_x2 = np.minimum(gt_box[2], pred_box[2])
                inter_y2 = np.minimum(gt_box[3], pred_box[3])

                inter_w = np.maximum(0, inter_x2 - inter_x1 + 1)
                inter_h = np.maximum(0, inter_y2 - inter_y1 + 1)
                inter_area = inter_w * inter_h

                gt_area = (gt_box[2] - gt_box[0] + 1) * (gt_box[3] - gt_box[1] + 1)
                pred_area = (pred_box[2] - pred_box[0] + 1) * (
                    pred_box[3] - pred_box[1] + 1
                )
                union_area = gt_area + pred_area - inter_area

                iou = (
                    (inter_area / union_area)
                    if (union_area > 0 and gt_box[0] > 0)
                    else 0.0
                )
                iou_matrix_frame[i, j] = iou

                # 计算 L1 距离
                l1_dist = np.sum(np.abs(pred_box - gt_box))
                l1_matrix_frame[i, j] = l1_dist

        # 使用匈牙利算法或贪婪匹配找到当前帧的最佳匹配
        if use_scipy:
            # Scipy 最小化代价，所以使用 -iou
            row_idx, col_idx = linear_sum_assignment(-iou_matrix_frame)
        else:
            # 贪婪匹配：优先匹配最高 IoU
            row_idx, col_idx = [], []
            matched_pred = np.zeros(num_valid_pred, dtype=bool)
            # 按 GT 索引迭代，为每个 GT 找到最佳未匹配的 Pred
            gt_order = np.argsort(
                -iou_matrix_frame.max(axis=1)
            )  # 优先考虑能获得高IoU的GT
            for r in gt_order:
                best_iou = -1
                best_c = -1
                # 寻找当前 GT 的最佳 Pred 匹配
                pred_order = np.argsort(-iou_matrix_frame[r, :])
                for c in pred_order:
                    if not matched_pred[c]:
                        if iou_matrix_frame[r, c] > best_iou:
                            best_iou = iou_matrix_frame[r, c]
                            best_c = c
                # 如果找到匹配
                if best_c != -1:
                    row_idx.append(r)
                    col_idx.append(best_c)
                    matched_pred[best_c] = True
            row_idx = np.array(row_idx)
            col_idx = np.array(col_idx)

        single_frame_permutation = np.arange(obj_num)
        # 将匹配结果存储到最终矩阵中
        for r, c in zip(row_idx, col_idx):
            single_frame_permutation[valid_gt_indices[r]] = valid_pred_indices[c]
            original_gt_idx = valid_gt_indices[r]
            matched_iou_matrix[original_gt_idx, f_idx] = iou_matrix_frame[r, c]
            matched_l1_matrix[original_gt_idx, f_idx] = l1_matrix_frame[r, c]

        permuted_bboxs[:, f_idx, :] = pred_bboxs[single_frame_permutation, f_idx, :]

        permutation.append(single_frame_permutation)

    permute_matrix = np.array(permutation).astype(np.int32)  # [frame_num， obj_num]
    # permuted_bboxs = np.transpose(np.transpose(pred_bboxs, (1, 0, 2))[np.arange(permute_matrix.shape[0])[:, None], permute_matrix], (1, 0, 2))

    # 对于没有匹配上的 GT 对象（在某些帧可能没有对应的 Pred），其 IoU 保持为 0，L1 保持为 -1

    return matched_iou_matrix, matched_l1_matrix, permute_matrix, permuted_bboxs


OUT_OF_GT = 0.2


def grounding2sam_process_reward(
    idx: int,
    raw_problem: str,
    inputs: list,
    grounding_indices: list,
    valid_mask: np.array,
    completions: list,
    arg_dict: list,
    **kwargs,
):
    masks = kwargs.get("masks", None)
    bboxs = kwargs.get("bboxs", None)
    points = kwargs.get("points", None)

    assert masks is not None and bboxs is not None and points is not None, (
        f"data items {masks if masks is None else bboxs if bboxs is None else points} should not be None"
    )

    # masks = torch.from_numpy(masks).float()
    assert bboxs.ndim == 3 and points.ndim == 3, (
        "bboxs and points should be 3D np.array of (obj_num, frame_num, 4)"
    )

    pred_bbox = [[] for _ in range(bboxs.shape[0])]
    scaled_bboxs = np.array(bboxs).astype(np.float32)
    offset = 0.0

    for i, (completion, data_info) in enumerate(zip(completions, arg_dict)):
        # completion = completion.strip()
        # print(f"completion: {completion}")

        img_resized_hw = data_info.get("img_resized_hw", None)[0]
        img_ori_hw = data_info.get("img_ori_hw", None)[0]
        if img_resized_hw is None or img_ori_hw is None:
            raise ValueError("img_resized_hw or img_ori_hw is None")

        h_ratio = img_resized_hw[0] / float(img_ori_hw[0])
        w_ratio = img_resized_hw[1] / float(img_ori_hw[1])

        # bboxs = np.stack(bboxs, axis=0)
        # points = np.stack(points, axis=0)
        scale = np.array([w_ratio, h_ratio, w_ratio, h_ratio]).astype(np.float32)

        # scaled_masks = torch.nn.functional.interpolate(
        #     masks, size=img_resized_hw, mode='nearest'
        # ).int().cpu().numpy()
        c_idx = grounding_indices[i]
        scaled_bboxs[:, c_idx, :] = bboxs[:, c_idx, :].astype(np.float32) * scale
        # scaled_points = points.astype(np.float32) * scale

        # ans = extract_answer(completion)
        ans = completion
        json_pattern = r"({[\s\S]*?})"
        js_match_list = re.findall(json_pattern, ans)

        cnt = 0
        for js in js_match_list:
            if cnt == len(pred_bbox):
                offset = offset - (len(js_match_list) - bboxs.shape[0])
                break
            try:
                js_result = json.loads(js)
                assert (
                    isinstance(js_result["bbox_2d"], list)
                    and len(js_result["bbox_2d"]) == 4
                ), "bbox_2d should have 4 elements"
                pred_bbox[cnt].append(js_result["bbox_2d"])
                cnt += 1
            except Exception:
                pred_bbox[cnt].append([-1, -1, -1, -1])
                cnt += 1
        if cnt < bboxs.shape[0]:
            for j in range(cnt, bboxs.shape[0]):
                pred_bbox[j].append([-1, -1, -1, -1])

        # pred_bboxes: np.ndarray [obj_num, frame_num, 4]

    array_indices = np.array(grounding_indices).astype(np.int32)

    # pred_bbox's shape is (obj_num, frame_num, 4)
    pred_bbox = np.array(pred_bbox).astype(np.float32)
    # gt scaled_bboxs's shape is (obj_num, frame_num, 4)
    scaled_bboxs = scaled_bboxs[:, array_indices, :]
    # get an IoU matrix, whose shape is (obj_num, frame_num) indicating
    # the max IoU over all IoUs between scaled_bboxs of that one frame of the same position

    # ious = iou(pred_bbox, scaled_bboxs)
    iou_matrix_frame, l1_matrix_frame, permute_matrix, permuted_pred_bboxs = (
        iou_framewise_object_matching(scaled_bboxs, pred_bbox)
    )

    # TODO: 使用最细粒度的有效IoU
    per_frame_per_obj_valid_mask = np.all(bboxs != -1, axis=2)[:, array_indices]

    # TODO: 原先有Bug，这里 valid_mask 对应原视频的有效帧，但是此处已经是 array_indices 筛选之后的。
    valid_mask = None
    valid_mask = per_frame_per_obj_valid_mask

    valid_ious = iou_matrix_frame[valid_mask]
    # valid_l1s = l1_matrix_frame[valid_mask]

    # selected_masks = (masks[:, array_indices, :] > 0).astype(np.int32)
    masks = (masks > 0).astype(np.int32)
    # gt_selected_mask_sum = np.sum(selected_masks, axis=(2, 3)) / np.prod(selected_masks.shape[-2:])  # 对所有对象、高度和宽度求和
    # FIXME: why 0,2,3
    gt_mask_sum = np.sum(
        masks, axis=(0, 2, 3)
    )  # 在帧数维度，对所有对象、高度和宽度求和
    gt_mask_ratio = gt_mask_sum / (np.max(gt_mask_sum) + 1e-6)
    gt_selected_mask_ratio = gt_mask_ratio[array_indices]

    # 有效 bbox 的索引
    valid_indices = np.argwhere(np.all(permuted_pred_bboxs != -1, axis=2))
    # 统计 unique frame 值
    pred_frame_valid_indices, frame_counts = np.unique(
        valid_indices[:, 1], return_counts=True
    )
    # 统计 unique obj 值
    pred_obj_valid_indices, obj_counts = np.unique(
        valid_indices[:, 0], return_counts=True
    )

    # 是否有多个帧，其中每一帧存在多于1个框（[-1, -1, -1, -1]为无效框）
    # is_multi_frame_and_multi_bbox = pred_frame_valid_indices.shape[0] > 1 and np.any(
    #     frame_counts > 1
    # )

    avg_iou = np.mean(valid_ious) if valid_ious.size > 0 else 0.0
    # avg_ratio = np.mean(gt_selected_mask_ratio) if gt_selected_mask_ratio.size > 0 else 0.0
    avg_ratio = (
        np.mean(gt_selected_mask_ratio) if gt_selected_mask_ratio.size > 0 else 0.0
    )

    offset = float((offset / (float(bboxs.shape[0]) + 1e-6)) * 0.5) / float(
        len(grounding_indices)
    )

    img_list = kwargs.get("frames", None)
    img_list = [pth(img).resolve() for img in img_list]
    video_dir = str(img_list[0].parent)
    frame_names = [
        str(img.resolve()) for img in img_list
    ]  # [f.name for f in video_dir.glob('*.jpg')] + [f.name for f in video_dir.glob('*.png')]

    # [ [obj_id, frame_id], [obj_id, frame_id], ... ]
    valid_indices = valid_indices.tolist()
    permuted_pred_bboxs_list = []

    for index in valid_indices:
        permuted_pred_bboxs_list.append(
            [
                int(index[0]),
                int(array_indices[index[1]]),
                (permuted_pred_bboxs[index[0], index[1], :] / scale).tolist(),
            ]
        )

    assert len(frame_names) == masks.shape[1], (
        f"frame_names and gt_masks should have the same length, {len(frame_names)} vs {masks.shape[1]}"
    )
    # print(f'boxs:{permuted_pred_bboxs_list}')
    call_request = {
        "has_valid_bbox": len(permuted_pred_bboxs_list) > 0,
        "request": {
            "video_dir": video_dir,
            "bbox": permuted_pred_bboxs_list,
            "frame_names": frame_names,
        },
        "object_num_mismatch": offset,
        "avg_iou": float(avg_iou),
        "avg_ratio": float(avg_ratio),
        "gt_masks": masks,
        "images": img_list,
    }
    # call_request =
    return call_request


def sam2_single_reward(
    response: Union[Dict | List],
    gt_masks: np.array,
    # images: list,
    # avg_iou: float,
    # avg_ratio: float,
    # object_num_mismatch: float,
    **kwargs,
):
    is_obj_as_new = not isinstance(response, dict)

    pred_mask = np.zeros_like(gt_masks)
    obj_pred_mask = []
    if not is_obj_as_new:
        for obj_id in response.keys():
            for frame_idx in response[obj_id].keys():
                if frame_idx < 0:
                    continue
                if frame_idx >= pred_mask.shape[1]:
                    # raise ValueError("frame_idx should be < pred_mask.shape[1]")
                    continue
                mask = response[obj_id][frame_idx].astype(np.uint8)
                resized_mask = cv2.resize(
                    mask[0],
                    (pred_mask.shape[3], pred_mask.shape[2]),
                    interpolation=cv2.INTER_NEAREST,
                )
                pred_mask[obj_id][frame_idx] = resized_mask

        overall_pred_mask = np.sum(np.transpose(pred_mask, (1, 0, 2, 3)), axis=1)

    else:
        for obj_mask in response:
            single_obj_mask = np.zeros_like(gt_masks[0])
            for frame_idx in range(pred_mask.shape[1]):
                mask = obj_mask[frame_idx].astype(np.uint8)
                resized_mask = cv2.resize(
                    mask[0],
                    (pred_mask.shape[3], pred_mask.shape[2]),
                    interpolation=cv2.INTER_NEAREST,
                )
                single_obj_mask[frame_idx] = resized_mask
            obj_pred_mask.append(np.array(single_obj_mask))
        obj_pred_mask = np.stack(obj_pred_mask, axis=0)
        overall_pred_mask = np.sum(np.transpose(obj_pred_mask, (1, 0, 2, 3)), axis=1)

    # for obj_id in range(pred_mask.shape[0]):
    #     for frame_idx in range(pred_mask.shape[1]):
    #         pred_mask[obj_id, frame_idx] = response[obj_id][frame_idx]

    overall_gt_mask = np.sum(np.transpose(gt_masks, (1, 0, 2, 3)), axis=1)

    final_metric = cal_metric(overall_pred_mask, overall_gt_mask)
    return final_metric


def sam2_gather_rewards(responses: list):
    """根据SAM2的响应计算奖励"""
    rewards = [sam2_single_reward(**response) for response in responses]
    # print(f"RANK[{torch.distributed.get_rank()}] SAM2 分割分数: {rewards}")
    return rewards


def cal_metric(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]):
    """_summary_

    Args:
        pred_masks (List[np.ndarray]): shaped as (n_frames, H, W)
        gt_masks (List[np.ndarray]): shaped as (n_frames, H, W)
    """
    j = db_eval_iou(gt_masks, pred_masks).mean()
    return j
    # f = db_eval_boundary(gt_masks, pred_masks).mean()
    # a = get_r2vos_accuracy(gt_masks, pred_masks).mean()


GAP_REWARD = 0.4
MASK_RATIO_REWARD = 1.0  # mask覆盖率奖励
INTERVAL_LENGTH_REWARD = 0.2  # 时间间隔长度奖励

MATCH_REWARD = 0.2  # 匹配奖励
ELEMENT_REWARD = 0.2  # start end description 都出现的奖励
TIME_CONSISTENCY_REWARD = 0.2  # 时间一致性奖励
OVER_LAP_PENELTY = 0.5  # 重叠惩罚

# 帧率参数，FPS=2，表示每秒两帧
FPS = 2


def time_reward_multi_time(completion: str, **kwargs):
    """评估视频时间间隔标记任务的奖励函数

    奖励条件:
    1. 回答的时间间隔内出现过mask，有奖励
    2. 间隔内出现mask的帧数比例越高，分数越高
    3. 时间间隔不能太长，必须在几张图片的数量之内，最好是4张以内

    Args:
        completion (str): 模型输出的答案字符串
        **kwargs: 附加参数，包含masks等信息

    Returns:
        float: 奖励值，范围[0, 1]
    """
    # 验证时间戳格式
    time_pattern = r"(\d+):(\d+)\.(\d+)"

    def parse_time_to_seconds(time_str):
        minutes, seconds, ms = map(int, re.match(time_pattern, time_str).groups())
        return minutes * 60 + seconds + ms * 10 / 1000

    cumulated_reward = 0.0
    time_piece = []

    json_pattern = r"({[\s\S]*?})"
    json_match = re.findall(json_pattern, completion)
    any_valid = False
    js_number = len(json_match)

    if js_number == 0:
        return cumulated_reward, None

    cumulated_reward += MATCH_REWARD * js_number

    for json_str in json_match:
        try:
            result = json.loads(json_str)

            # 提取时间戳
            start_time = result.get("start_time", None)
            end_time = result.get("end_time", None)
            description = result.get("description", None)

            start_match = re.match(time_pattern, start_time)
            end_match = re.match(time_pattern, end_time)

            if not (
                start_match is not None
                and end_match is not None
                and description is not None
            ):
                continue

            cumulated_reward += ELEMENT_REWARD

            # 解析时间戳为秒数
            start_seconds = parse_time_to_seconds(start_time)
            end_seconds = parse_time_to_seconds(end_time)

            # 计算开始和结束的帧索引
            frame_count = len(kwargs.get("frames", []))
            total_duration = frame_count / FPS

            if start_seconds > end_seconds or end_seconds > total_duration:
                continue

            cumulated_reward += TIME_CONSISTENCY_REWARD

            time_piece.append((start_seconds, end_seconds, description))
            any_valid = True

        except Exception:
            continue

    if not any_valid:
        return cumulated_reward / js_number, None

    valid_frame_idx = []
    for start, end, des in time_piece:
        # 将时间转为帧索引
        start_frame_idx = min(int(start * FPS), frame_count - 1)
        end_frame_idx = min(int(end * FPS), frame_count - 1)
        duration = np.arange(start_frame_idx, end_frame_idx + 1)
        valid_frame_idx.append(duration)

    # 获取masks信息
    masks = kwargs.get("masks")
    if masks is None:
        raise ValueError("masks is None")

    # 计算时间间隔内每一帧是否有mask
    masks_sum = np.sum(masks, axis=(2, 3))  # 对所有对象、高度和宽度求和

    # FIXME: 这里假设masks的形状为 [obj_num, frame_num, H, W], obj_num == 1

    ratio_rewards, length_rewards = [], []

    for duration in valid_frame_idx:
        interval_frames = duration.shape[0]
        # 计算有多少帧包含mask (非零值)
        selected_masks = masks_sum[:, duration]
        frames_with_mask = np.count_nonzero(selected_masks > 0)

        # 条件2: 时间间隔内有mask的帧数比例
        mask_ratio = (
            (frames_with_mask / (interval_frames * selected_masks.shape[0]))
            if interval_frames > 0
            else 0
        )
        mask_ratio_reward = MASK_RATIO_REWARD * mask_ratio

        # TODO: 强制限制单段关键帧长度为 1 帧
        length_reward_o_penalty = (1 - interval_frames) * INTERVAL_LENGTH_REWARD
        length_reward_o_penalty = (
            length_reward_o_penalty if length_reward_o_penalty < 0 else 0
        )

        ratio_rewards.append(mask_ratio_reward)
        length_rewards.append(length_reward_o_penalty)

    all_indices = np.concatenate(valid_frame_idx)
    over_lap = len(all_indices) - len(np.unique(all_indices))
    over_lap_penelty = over_lap * OVER_LAP_PENELTY

    cumulated_reward += sum(ratio_rewards)
    cumulated_reward = cumulated_reward / js_number

    # TODO: 不再平均关键帧段数，直接计算总量
    total_overlap_reward = sum(length_rewards) - over_lap_penelty

    # TODO: 帧分布上的奖励
    N = 4
    all_g_indices = np.unique(all_indices)
    target_frame_number_gap = N - all_g_indices.shape[0]
    if target_frame_number_gap >= 0:
        total_frame_reward = all_g_indices.shape[0] * GAP_REWARD
    else:
        total_frame_reward = (N + target_frame_number_gap - 1) * GAP_REWARD

    # 目前的全部奖励：
    # mask_ratio 1.0
    # element_reward 0.5
    # match 0.5
    # time_consistency 0.5
    # 总数 2.5
    # overlap frames * (-0.5)
    # 总帧数奖励 5 * 0.4 = 2.0
    # 最好最好目前是 4.5 的分数
    # sam 1 + 2 * iou + offset
    # 目前最高是 7.5

    g_indices = all_g_indices[:N]
    selected_masks = masks_sum[:, g_indices]
    g_indices = g_indices.tolist()
    description_list = []

    for idx in g_indices:
        for j, duration in enumerate(valid_frame_idx):
            if np.any(np.isin(idx, duration)):
                description_list.append(time_piece[j][2])
                break

    reward_dict = {
        "cumulated_reward": cumulated_reward,
        "mask_ratio_reward": ratio_rewards,
        "interval_length_reward": length_rewards,
        "grounding_indices": g_indices,
        # TODO: make sure valid json is alway score higher than non-valid
        "total_frame_reward": total_frame_reward,
        "total_overlap_reward": total_overlap_reward,
        "grounding": {
            "grounding_indices": g_indices,
            "descriptions": description_list,
            "valid_mask": selected_masks > 0,
        },
        "kwargs": kwargs,
    }

    return cumulated_reward, reward_dict
