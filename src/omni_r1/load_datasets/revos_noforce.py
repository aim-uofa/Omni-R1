import copy
import json
import logging
import os
import random
import re
import shutil
from pathlib import Path as pth
from typing import Dict, List, Tuple, Union

import diskcache as dc
import numpy as np
import pycocotools.mask as maskUtils
import torch
from datasets import Dataset
from PIL import Image
from scipy.ndimage import distance_transform_edt

from omni_r1.eval_utils import db_eval_boundary, db_eval_iou, get_r2vos_accuracy

# from qwen_omni_utils.v2_5.vision_process import smart_resize, VIDEO_MAX_PIXELS, VIDEO_MIN_PIXELS


QUESTION_TEMPLATE = (
    'Given a video containing {frames} frames, please find the objects described as "{Question}" in these video frames with bbox and points.'
    "Compare all frames and identify each target object consistently."
    "Output your thinking process in <think> </think> tags and final grounding results in <answer> </answer> tags."
    "Output the one bbox of each interested object for each frame using JSON format."
    "The grounding result should be in the format of json. 0, 1,..., frame_num-1 indicate frame idx."
    "If the object is not found in the frame, please output a dummy bbox as [-1, -1, -1, -1]"
    "e.g. if only one object is found in a two frame video, you should response as: {AnswerOne}."  # \n If two or more objects found: {AnswerMore}."
)

# VIDEO_TIMESTAMP_TEMPLATE = (
#     "Given a video containing [frames] frames and a reference instruction: [ref_prompt] that may involve temporal behavior, "
#     # "identify when the object or action described in the instruction [ref_prompt] first appears and when it disappears from the video. "
#     "identify the exact object [ref_prompt] that matches the description in the video. "
#     "Select the most relevant 1 second duration or less that contains the referred object and simple to recognize it. "
#     "Then, simplify the identified object into a short and clear visual grounding description that can be used for single-image understanding. "
#     "Avoid temporal phrases and comparison phrases like 'walking', 'moving', 'approaching', 'bigger' or 'smaller', and instead describe visible visual cues like clothing, pose, position, or grouping. "
#     "Explain your reasoning in <think>...</think> and output the final result in <answer>...</answer>. "
#     "Note that you do not locate from beginning, but the most relevant duration of 0.5-2.0 seconds or multiple small durations of highly representative moments. "
#     "Your final answer should be a JSON object in the following format:\n"
#     "<think> think process here </think>\n"
#     "<answer>\n"
#     "{\n"
#     "  \"start_time\": \"00:0[start]\","
#     "  \"end_time\": \"00:0[end]\", "
#     "  \"description\": \"direct description of referred object\"\n"
#     "}\n"
#     "//another json object if necessary\n"
#     "</answer>\n"
# )

# 新增的视频时间标记prompt模板
VIDEO_TIMESTAMP_TEMPLATE = (
    "Given a [frames] seconds video and a reference instruction: [ref_prompt] that may involve temporal behavior, "
    # "identify when the object or action described in the instruction [ref_prompt] first appears and when it disappears from the video. "
    "identify the exact object(s) [ref_prompt] in the video that matches the description. "
    "Select the most relevant duration that contains the referred object(s) with the best view. "
    "Then, simplify the identified object into a short and clear visual grounding description that can be used for single-image reference. "
    "Avoid temporal phrases and comparison phrases like 'walking', 'moving', 'approaching', 'bigger' or 'smaller', but instead describe visible visual cues like clothing, pose, position, or grouping. "
    "Explain your reasoning in <think>...</think> and output the final result in <answer>...</answer>. "
    "Note that you do not locate from beginning, but the most relevant one duration of 0.5-2.0 seconds or multiple small durations of highly representative moments. "
    "Your final answer should be a JSON object in the following format:\n"
    "<think> think process here </think>\n"
    "<answer>\n"
    "{\n"
    '  "start_time": "00:[start]",'
    '  "end_time": "00:[end]",'
    '  "description": "direct description of referred object(s)"\n'
    "}\n"
    "//another json object if necessary\n"
    "{\n"
    '  "start_time": "00:[start1]",'
    '  "end_time": "00:[end1]",'
    '  "description": "direct description of referred object(s)"\n'
    "}\n"
    "</answer>\n"
)

# JSON_EXAMPLE = (
#      "{ '0': { 'bbox': [10, 40, 100, 150], 'points_1': [50, 130], 'points_2': [80, 120],},'1': {'bbox': [60, 90, 180, 230],'points_1': [150, 190],'points_2': [140, 200]},...}"
# )

JSON_EXAMPLE = (
    '{ "0": { "bbox_2d": [10, 40, 100, 150]},"1": {"bbox_2d": [60, 90, 180, 230]}}'
)

ANSWER_ONE = "<think> think process here </think> <answer> object_1: {json_example}, object_2(if there exists): [the same json format] </answer>"

# ANSWER_MORE = (
#     "<think> think process here </think>"
#     "<answer> { object_1: {json_example}, object_2: {json_example} } </answer>"
# )

# SEG_QUESTIONS = [
#     "Can you ground the objects described as '{class_name}' in this video?",
#     "Please ground {class_name} in these video frames.",
#     "What is {class_name} in this video? Please respond with json format grounding result.",
#     "Can you track and segment the {class_name} across these video frames?",
#     "Please identify and segment the {class_name} throughout this video sequence.",
#     "Where is the {class_name} in these frames? Please respond with bbox and points.",
#     "Can you highlight the {class_name} in all frames with json bbox and points?",
# ]

# GROUNDING_TEMPLATE = (
#     "Please find '{Question}' with bbox and points."
#     "Compare the difference between objects and find the most closely matched one."
#     "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
#     "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format."
#     "i.e., <think> thinking process here </think>"
#     "<answer>{Answer}</answer>"
# )

# grounding_answer = (
#     "{ \"bbox\" : [10,100,200,210], \"points_1\" : [30,110], \"points_2\" : [35,180] }"
# )

do_thinking = True


ORIGINAL_GROUNDING_TEMPLATE = (
    "Please find '{Question}' with bbox."
    "Compare the difference between objects and find the most closely matched one."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the bbox of the most interested object in JSON format."
    "i.e., <think> thinking process here </think>"
    "<answer>{Answer}</answer>"
)

origin_grounding_answer = '{ "bbox_2d" : [x1,y1,x2,y2] }'

NOTHINKING_GROUNDING_TEMPLATE = (
    "Please find '{Question}' with bbox."
    "Compare the difference between objects and find the most closely matched one."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the bbox of the most interested object in JSON format."
    "i.e., <think> thinking process here </think>"
    "<answer>{Answer}</answer>"
)


class VideoReVOSDataset(Dataset):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        dataset_path: Union[str, pth],
        select_number=1,
        use_all_frames=True,
        sampled_frames=6,
        min_frames=8,
        max_frames=24,
        image_size=(448, 448),
        use_timestamp_prompt=True,  # 新增参数：是否使用时间戳提示
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
        print(
            f"################# Object Number up to: {self.select_number} ################# "
        )
        self.use_all_frames = use_all_frames
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.sampled_frames = 2 if os.getenv("DEBUG_MODE") == "true" else sampled_frames
        print(
            f"################# min_frames: {self.min_frames}, max_frames: {self.max_frames} ################# "
        )
        self.image_size = image_size
        self.use_timestamp_prompt = use_timestamp_prompt  # 记录是否使用时间戳prompt

        self.lazy = False
        # self.transformer = T.Compose([
        #     T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        #     T.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
        #     T.ToTensor(),
        #     T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD)
        # ])

        # 仅加载元数据，不预处理
        self.load_dataset()

        pass
        # 创建视频索引用于__getitem__
        # self.video_indices = list(range(len(self.videos)))

    def get_supplementary_dataset(self):
        expression_file = "revos_supplementary_train_.json"

        return VideoReVOSDataset(
            dataset_path=self.dataset_folder,
            image_folder=self.image_folder,
            expression_file=expression_file,
            mask_file=self.mask_file,
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
                'data_type': ['video'] * batch_size,
                'problem_type': ['video-segmentation'] * batch_size,
                'problem_id': batch list of video_meta['anno_id'], reassigned in creating process, not the original anno_id in meta_expression
                'frames': batch list of frame image path, [[batch_0_frame_list], [batch_1_frame_list], ...],
                'masks': obj_i as video_masks object-wise as np.array (frames, H, W), in batch list [b1, b2, ...], bi: [obj1, obj2, ...]
                'obj_number': number of objects in each video, list of int, [obj1_number, obj2_number, ...]
                'bboxs': same of structure of 'masks', shaped as (frames, 4)
                'points': same of structure of 'masks', shaped as (frames, 4)
                'expression': batch list of expression str,
                'prompt': prompt list of dict, refer to _create_prompt()

            }
        """
        # # 如果在分布式环境中，根据 rank 和 world_size 对数据集进行分片访问
        # local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # world_size = int(os.environ.get("WORLD_SIZE", 1))

        # if world_size > 1:
        #     # 对总索引进行分片
        #     num_samples = len(self.videos)
        #     indices = list(range(local_rank, num_samples, world_size))
        #     for i , index in enumerate(idx):
        #         if index >= len(indices):
        #             # 如果索引超出了分片大小，可以循环采样或抛出异常
        #             index = index % len(indices)
        #         idx[i] = indices[index]

        # idx = idx % len(self.videos)  # 确保索引在范围内
        idx = idx if isinstance(idx, list) else [idx]
        bs = len(idx)
        video_names = [self.videos[i] for i in idx]
        rets = []
        for video_name in video_names:
            # 获取视频对应的expression meta index list
            video_objects = self.vid2metaid[video_name]

            video_meta = []
            # 获取选定对象的元数据, 存在一些meta项，object_id为空，跳过
            for i in video_objects:
                if len(self.metas[i]["obj_id"]) == 0:
                    continue
                video_meta.append(self.metas[i])
            rets.append(self._process_single_item(video_meta))

        ret = {}
        for k in rets[0].keys():
            for i in range(bs):
                if k not in ret:
                    ret[k] = []
                ret[k].append(rets[i][k])
        return rets[0]

    def _process_single_item(self, video_meta):
        """处理单个视频数据项"""
        # random pick one expression
        selected_meta = random.choice(video_meta)

        if self.use_all_frames:
            # 采用宽松的视频帧处理，尽量多些帧数
            selected_frame_indices = list(range(len(selected_meta["frames"])))
            if len(selected_frame_indices) > self.min_frames:
                f_num = random.randint(self.min_frames, self.max_frames)
                selected_frame_indices = np.linspace(
                    0, len(selected_meta["frames"]) - 1, f_num, dtype=int
                ).tolist()
        else:
            # 随机选择帧
            len_frames = len(selected_meta["frames"])
            selected_frame_indices = np.linspace(
                0, len_frames - 1, self.sampled_frames, dtype=int
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
        full_video_masks = []
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
        if self.use_timestamp_prompt:
            # FIXME: temporarily disable timestamp prompt
            start = random.randint(0, len(frames) - 2)
            end = (min(start + random.randint(1, 4), len(frames) - 1)) * 0.5
            start = start * 0.5

            start1 = random.randint(0, len(frames) - 2)
            end1 = (min(start1 + random.randint(1, 4), len(frames) - 1)) * 0.5
            start1 = start1 * 0.5

            content.append(
                {
                    "type": "text",
                    "text": VIDEO_TIMESTAMP_TEMPLATE.replace(
                        "[ref_prompt]", "[" + expression + "]"
                    )
                    .replace("[frames]", str(len(frames) / 2))
                    .replace("[start]", f"{start:05.2f}")
                    .replace("[end]", f"{end:05.2f}")
                    .replace("[start1]", f"{start1:05.2f}")
                    .replace("[end1]", f"{end1:05.2f}"),
                }
            )
            # content.append({
            #     "type": "text",
            #     "text": QUESTION_TEMPLATE.format(Question=expression, AnswerOne=ANSWER_ONE.format(json_example=JSON_EXAMPLE))#, AnswerMore=ANSWER_MORE.format(json_example=JSON_EXAMPLE))
            # })
        else:
            content.append(
                {
                    "type": "text",
                    "text": QUESTION_TEMPLATE.format(
                        Question=expression,
                        frames=len(frames),
                        AnswerOne=ANSWER_ONE.format(json_example=JSON_EXAMPLE),
                    ),  # , AnswerMore=ANSWER_MORE.format(json_example=JSON_EXAMPLE))
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
    first_val = dist_map[first_yx]

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
    second_val = dist_map_masked[second_yx]

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


# BBOX_FORMAT_REWARD = 0.25
# POINTS_FORMAT_REWARD = 0.25
# FOUND_REWARD = 0.75


# def reward_assign(
#         object_number_match: bool,
#         json_loadable: bool,
#         valid_format_array: np.array,
#         format_reward: np.array = None,
#         consistency: np.array = None,
#         points_in_mask: np.array = None,
#         points_l1: np.array = None,
#         box_iou: np.array = None,
#         box_l1: np.array = None,
#     ):
#     """assigning reward according to evaluation metrics

#     Args:
#         object_number_match (bool): bool
#         json_loadable (bool): bool
#         valid_format_array (np.array): [obj_num, frame_num]
#         format_reward (np.array): [obj_num, frame_num]
#         consistency (np.array): [obj_num, frame_num]
#         points_in_mask (np.array): [obj_num, frame_num]
#         points_l1 (np.array): [obj_num, frame_num]
#         box_iou (np.array): [obj_num, frame_num]
#         box_l1 (np.array): [obj_num, frame_num]
#     """
#     valid_format_reward = valid_format_array.sum()/np.prod(valid_format_array.shape) # 1, actually
#     if not valid_format_array.all():
#         return valid_format_reward, {"valid_format": valid_format_reward}

#     print("format correct...")

#     avg_format_reward = format_reward.sum()/np.prod(valid_format_array.shape)
#     # consistency_reward = consistency.sum()/np.prod(valid_format_array.shape)
#     # points_in_mask_reward = points_in_mask.sum()/np.prod(valid_format_array.shape)
#     # points_l1_reward = (points_l1 < 50).astype(np.int32).sum()/np.prod(valid_format_array.shape)
#     box_iou_reward = (box_iou > 0.5).astype(np.int32).sum()/np.prod(valid_format_array.shape)

#     # box_l1_reward = (box_l1 < 10).astype(np.int32).sum()/np.prod(valid_format_array.shape)

#     if not (avg_format_reward == FOUND_REWARD + BBOX_FORMAT_REWARD).all():
#         return valid_format_reward + avg_format_reward, {"valid_format": valid_format_reward, "avg_format": avg_format_reward, "box_iou": box_iou_reward} # + consistency_reward + points_in_mask_reward + points_l1_reward + box_l1_reward

#     if not (box_iou > 0.5).all():
#         return valid_format_reward + avg_format_reward + box_iou_reward, {"valid_format": valid_format_reward, "avg_format": avg_format_reward, "box_iou": box_iou_reward} # + consistency_reward + points_in_mask_reward + points_l1_reward + box_l1_reward

#     return valid_format_reward + avg_format_reward + box_iou_reward + (box_iou.sum()/np.prod(valid_format_array.shape)), {"valid_format": valid_format_reward, "avg_format": avg_format_reward, "box_iou": box_iou_reward + (box_iou.sum()/np.prod(valid_format_array.shape))} # + consistency_reward + points_in_mask_reward + points_l1_reward + box_l1_reward

# def calculate_reward(
#     object_number_match: bool,
#     json_loadable: bool,
#     valid_format_array: np.array,
#     # valid_frame_mask: np.array,
#     format_reward: np.array,
#     # box_iou: np.array,
#     # box_l1: np.array,
#     # consistency: np.array,
#     # points_in_mask: np.array,
#     # points_l1: np.array
#     obj_number: int,
#     masks: np.array,
#     gt_bboxs: np.array,
#     gt_points: np.array,
#     pred_datas: List[dict],
# ) -> tuple:
#     """_summary_

#     Args:
#         object_number_match (bool): if object number and json format substr number matches
#         json_loadable (bool): if all the json str are loadable
#         valid_format_array (np.array): np.array (obj_num,), whether the json format of each object is valid as definition
#         format_reward (np.array): np.array (obj_num, frame_num), the format reward of each frame for each object
#         obj_number (int): gt object number
#         masks (np.array): _description_
#         gt_bboxs (np.array): _description_
#         gt_points (np.array): _description_
#         pred_datas (List[dict]): _description_

#     Returns:
#         float: final reward
#     """

#     # def cal_box_iou(box1, box2):
#     #     """计算两个边界框的IoU"""
#     #     inter_x1 = max(box1[0], box2[0])
#     #     inter_y1 = max(box1[1], box2[1])
#     #     inter_x2 = min(box1[2], box2[2])
#     #     inter_y2 = min(box1[3], box2[3])

#     #     if inter_x1 < inter_x2 and inter_y1 < inter_y2:
#     #         inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
#     #     else:
#     #         inter = 0

#     #     area1 = (box1[2]-box1[0]+1)*(box1[3]-box1[1]+1)
#     #     area2 = (box2[2]-box2[0]+1)*(box2[3]-box2[1]+1)
#     #     union = area1 + area2 - inter

#     #     return float(inter)/union

#     # def cal_box_l1_distance(box1, box2):
#     #     """计算两个边界框的L1距离"""
#     #     return (abs(box1[0]-box2[0]) + abs(box1[1]-box2[1]) + abs(box1[2]-box2[2]) + abs(box1[3]-box2[3])) / 4

#     # def cal_points_distance(points1, points2):
#     #     """计算两组点之间的距离"""
#     #     dist1 = math.sqrt((points1[0][0]-points2[0][0])**2 + (points1[0][1]-points2[0][1])**2) + \
#     #             math.sqrt((points1[1][0]-points2[1][0])**2 + (points1[1][1]-points2[1][1])**2)

#     #     dist2 = math.sqrt((points1[0][0]-points2[1][0])**2 + (points1[0][1]-points2[1][1])**2) + \
#     #             math.sqrt((points1[1][0]-points2[0][0])**2 + (points1[1][1]-points2[0][1])**2)

#     #     return min(dist1, dist2) / 2

#     def is_points_in_bboxs(points, bboxs):
#         """检查点是否在边界框内"""
#         # return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]
#         # 构建条件掩码：检查点是否在边界框内
#         # points 形状: [obj_num, frame_num, 4] (其中4个值是[x1, y1, x2, y2])
#         # bboxs 形状: [obj_num, frame_num, 4] (其中4个值是[x1, y1, x2, y2])

#         # 检查 points_1 是否在边界框内
#         points1_in_box = np.logical_and(
#             np.logical_and(
#                 points[:, :, 0] >= bboxs[:, :, 0],  # x1 >= bbox_x1
#                 points[:, :, 1] >= bboxs[:, :, 1]   # y1 >= bbox_y1
#             ),
#             np.logical_and(
#                 points[:, :, 0] <= bboxs[:, :, 2],  # x1 <= bbox_x2
#                 points[:, :, 1] <= bboxs[:, :, 3]   # y1 <= bbox_y2
#             )
#         )

#         # 检查 points_2 是否在边界框内
#         points2_in_box = np.logical_and(
#             np.logical_and(
#                 points[:, :, 2] >= bboxs[:, :, 0],  # x2 >= bbox_x1
#                 points[:, :, 3] >= bboxs[:, :, 1]   # y2 >= bbox_y1
#             ),
#             np.logical_and(
#                 points[:, :, 2] <= bboxs[:, :, 2],  # x2 <= bbox_x2
#                 points[:, :, 3] <= bboxs[:, :, 3]   # y2 <= bbox_y2
#             )
#         )

#         # 两个点都在边界框内时，一致性为 True
#         return np.logical_and(points1_in_box, points2_in_box).astype(np.float32)


#     def is_points_in_mask(masks: np.array, points: np.array, gt_points: np.array) -> Tuple[np.array, np.array]:
#         """检查点是否在掩码内，使用NumPy向量化操作加速

#         Args:
#             masks (List[np.array]): 掩码列表，每个元素是一个对象的所有帧的掩码，形状为 [frame_num, H, W]
#             points (np.array): 点坐标，形状为 [obj_num, frame_num, 4]，每个点包含 [x1, y1, x2, y2]

#         Returns:
#             np.array: 点是否在掩码内的结果，形状为 [obj_num, frame_num, 2]
#             np.array: points的L1距离，形状为 [obj_num, frame_num]
#         """
#         # 获取对象数量和帧数
#         obj_num = points.shape[0]
#         frame_num = points.shape[1]
#         H, W = masks.shape[2], masks.shape[3]  # 掩码形状为 [obj_num, frame_num, H, W]

#         # 初始化结果数组，形状为 [obj_num, frame_num, 2]
#         in_mask = np.zeros((obj_num, frame_num, 2), dtype=np.float32)

#         # 提取点的坐标
#         x1 = points[:, :, 0].astype(np.int32)  # [obj_num, frame_num]
#         y1 = points[:, :, 1].astype(np.int32)
#         x2 = points[:, :, 2].astype(np.int32)
#         y2 = points[:, :, 3].astype(np.int32)

#         gx1 = gt_points[:, :, 0].astype(np.int32)  # [obj_num, frame_num]
#         gy1 = gt_points[:, :, 1].astype(np.int32)
#         gx2 = gt_points[:, :, 2].astype(np.int32)
#         gy2 = gt_points[:, :, 3].astype(np.int32)

#         # Calculate L1 distance between predicted points and ground truth points
#         # We need to consider both possible assignments: (p1,gp1,p2,gp2) and (p1,gp2,p2,gp1)
#         l1_dist_assignment1 = np.abs(x1 - gx1) + np.abs(y1 - gy1) + np.abs(x2 - gx2) + np.abs(y2 - gy2)
#         l1_dist_assignment2 = np.abs(x1 - gx2) + np.abs(y1 - gy2) + np.abs(x2 - gx1) + np.abs(y2 - gy1)

#         # Take the minimum distance (optimal assignment)
#         points_l1 = np.minimum(l1_dist_assignment1, l1_dist_assignment2)


#         # 创建掩码，检查点是否有效（非-1）以及是否在图像范围内
#         valid_mask1 = np.logical_and(
#             np.logical_and(x1 >= 0, y1 >= 0),
#             np.logical_and(x1 < W, y1 < H)
#         )

#         valid_mask2 = np.logical_and(
#             np.logical_and(x2 >= 0, y2 >= 0),
#             np.logical_and(x2 < W, y2 < H)
#         )

#         # 获取空掩码的掩码, [obj_num, frame_num], True for empty
#         empty_masks = np.sum(masks, axis=(2,3)) == 0 #np.array([[mask.sum() == 0 for mask in obj_masks] for obj_masks in masks])

#         # 对每个对象和每一帧进行处理
#         for obj_idx in range(obj_num):
#             for frame_idx in range(frame_num):
#                 # 当前帧掩码为空的情况
#                 if empty_masks[obj_idx, frame_idx]:
#                     # 如果掩码为空且点是(-1,-1)，则认为点在掩码内
#                     if x1[obj_idx, frame_idx] == -1 and y1[obj_idx, frame_idx] == -1:
#                         in_mask[obj_idx, frame_idx, 0] = 1.0
#                     if x2[obj_idx, frame_idx] == -1 and y2[obj_idx, frame_idx] == -1:
#                         in_mask[obj_idx, frame_idx, 1] = 1.0
#                     continue

#                 # 检查第一个点是否在掩码内
#                 if valid_mask1[obj_idx, frame_idx]:
#                     x, y = x1[obj_idx, frame_idx], y1[obj_idx, frame_idx]
#                     if masks[obj_idx, frame_idx, y, x] > 0:
#                         in_mask[obj_idx, frame_idx, 0] = 1.0

#                 # 检查第二个点是否在掩码内
#                 if valid_mask2[obj_idx, frame_idx]:
#                     x, y = x2[obj_idx, frame_idx], y2[obj_idx, frame_idx]
#                     if masks[obj_idx, frame_idx, y, x] > 0:
#                         in_mask[obj_idx, frame_idx, 1] = 1.0

#         return np.sum(in_mask, axis=-1), points_l1


#     def object_matching(gt_bboxs: np.array, pred_bboxs: np.array) -> Tuple[np.array, np.array]:
#         """
#         为GT masks和预测边界框找到最佳的对象级别匹配，使用NumPy向量化计算IoU

#         Args:
#             gt_bboxs: GT bbox列表，每个元素是一个对象的所有帧的掩码包围盒，形状为 [obj, frame_num, 4]
#             pred_bboxs: 预测的边界框，形状为 [obj_num, frame_num, 4]

#         Returns:
#             np.array: 最优匹配的排列索引，可用于重新排列预测对象以匹配GT对象
#             np.array: IoU矩阵，形状为 [obj_num, obj_num, frame]
#             np.array: l1矩阵，形状为 [obj_num, obj_num, frame]
#         """
#         assert pred_bboxs.ndim == 3, "pred_bboxs should be 3D np.array of (obj_num, frame_num, 4)"
#         assert gt_bboxs.shape[0] == pred_bboxs.shape[0], "gt_bboxs and pred_bboxs should have the same number of objects"
#         assert gt_bboxs.ndim == 3, "gt_bboxs should be 2D np.array of (obj_num, frame_num, 4)"

#         # 获取对象数量和帧数
#         obj_num, frame_num = gt_bboxs.shape[0], gt_bboxs.shape[1]

#         # gt_bboxs = np.stack(gt_bboxs, axis=0)  # [obj_num, frame_num, 4]

#         # 计算IoU矩阵：每个GT对象与每个预测对象之间的IoU
#         # 形状为 [gt_obj_num, pred_obj_num]
#         iou_matrix = np.zeros((obj_num, obj_num), dtype=np.float32)
#         iou_matrix_frame = np.zeros((obj_num, obj_num, frame_num), dtype=np.float32)
#         l1_matrix_frame = np.zeros((obj_num, obj_num, frame_num), dtype=np.float32)

#         # 使用向量化操作计算所有对象对之间的IoU
#         for gt_idx in range(obj_num):
#             for pred_idx in range(obj_num):
#                 # 提取当前对象对的所有帧的边界框
#                 gt_boxes = gt_bboxs[gt_idx]   # [frame_num, 4]
#                 pred_boxes = pred_bboxs[pred_idx]  # [frame_num, 4]

#                 # 创建有效帧掩码（非dummy框）
#                 valid_mask = np.logical_and(
#                     ~np.all(gt_boxes == -1, axis=-1),
#                     ~np.all(pred_boxes == -1, axis=-1)
#                 )

#                 if not np.any(valid_mask):
#                     continue

#                 # # 只选择有效帧
#                 # valid_gt_boxes = gt_boxes[valid_mask]    # [valid_frames, 4]
#                 # valid_pred_boxes = pred_boxes[valid_mask]  # [valid_frames, 4]

#                 # 计算交叉区域的坐标
#                 inter_x1 = np.maximum(gt_boxes[:, 0], pred_boxes[:, 0])
#                 inter_y1 = np.maximum(gt_boxes[:, 1], pred_boxes[:, 1])
#                 inter_x2 = np.minimum(gt_boxes[:, 2], pred_boxes[:, 2])
#                 inter_y2 = np.minimum(gt_boxes[:, 3], pred_boxes[:, 3])

#                 # 计算交叉区域的宽度和高度
#                 inter_w = np.maximum(0, inter_x2 - inter_x1 + 1)
#                 inter_h = np.maximum(0, inter_y2 - inter_y1 + 1)

#                 # 计算交叉区域的面积
#                 inter_area = inter_w * inter_h

#                 # 计算各自边界框的面积
#                 gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * \
#                         (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
#                 pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0] + 1) * \
#                             (pred_boxes[:, 3] - pred_boxes[:, 1] + 1)

#                 # 计算并集面积
#                 union_area = gt_area + pred_area - inter_area

#                 # 计算IoU
#                 ious = inter_area / union_area

#                 # 将逐帧IoU存储
#                 iou_matrix_frame[gt_idx, pred_idx] = ious

#                 # 计算平均IoU
#                 if len(ious) > 0:
#                     iou_matrix[gt_idx, pred_idx] = np.mean(ious[valid_mask])

#                 l1_matrix_frame[gt_idx, pred_idx] = np.sum(np.abs(pred_boxes - gt_boxes), axis=-1)

#         # 使用匈牙利算法找到最佳匹配
#         # 由于匈牙利算法是最小化代价，我们需要将IoU取负（最大化IoU等于最小化-IoU）
#         try:
#             from scipy.optimize import linear_sum_assignment
#             row_idx, col_idx = linear_sum_assignment(-iou_matrix)
#             sort_idx = np.argsort(row_idx)
#             permutation_idx = col_idx[sort_idx]
#         except ImportError:
#             # 如果scipy不可用，则使用贪婪匹配
#             # 优先匹配具有最高IoU的对象
#             print('scipy not available, using greedy matching')
#             permutation_idx = np.zeros(obj_num, dtype=np.int32)
#             used_pred = np.zeros(obj_num, dtype=bool)

#             for gt_idx in range(obj_num):
#                 best_iou = -1
#                 best_pred = -1
#                 for pred_idx in range(obj_num):
#                     if used_pred[pred_idx]:
#                         continue
#                     if iou_matrix[gt_idx, pred_idx] > best_iou:
#                         best_iou = iou_matrix[gt_idx, pred_idx]
#                         best_pred = pred_idx

#                 if best_pred >= 0:
#                     permutation_idx[gt_idx] = best_pred
#                     used_pred[best_pred] = True

#         return permutation_idx, iou_matrix_frame, l1_matrix_frame


#     # frame_num = valid_frame_mask.shape[0]
#     # assert valid_frame_mask.shape\
#     #         == format_reward.shape\
#     #         == box_iou.shape\
#     #         == box_l1.shape\
#     #         == consistency.shape\
#     #         == points_in_mask.shape\
#     #         == points_l1.shape, "all rewards should have the same shape"

#     if not valid_format_array.all():
#         # FIXME: if not all objects are format-correct, return format reward only
#         return reward_assign(
#             object_number_match=object_number_match,
#             json_loadable=json_loadable,
#             valid_format_array=valid_format_array
#         )

#     pred_bboxs: np.array = np.zeros((format_reward.shape[0], format_reward.shape[1], 4))
#     # pred_points: np.array = np.zeros((format_reward.shape[0], format_reward.shape[1], 4))

#     # consistency: np.array = np.zeros_like(format_reward)
#     # points_in_mask: np.array = np.zeros_like(format_reward)
#     # points_l1: np.array = np.zeros_like(format_reward).astype(np.uint32)
#     # box_iou: np.array = np.zeros_like(format_reward).astype(np.float32)
#     # box_l1: np.array = np.zeros_like(format_reward).astype(np.uint32)

#     for obj, pred_data in enumerate(pred_datas):
#         for f, frame_data in pred_data.items():
#             frame_idx = int(f)
#             pred_bboxs[obj][frame_idx] = np.array(frame_data['bbox'])
#             # pred_points[obj][frame_idx][:2] = np.array(frame_data['points_1'])
#             # pred_points[obj][frame_idx][2:] = np.array(frame_data['points_2'])

#     # consistency = is_points_in_bboxs(pred_points, pred_bboxs)
#     permutation_idx, box_iou_matrix, box_l1_matrix = object_matching(gt_bboxs, pred_bboxs)

#     matched_box_iou_matrix = np.einsum('iij...->i...', box_iou_matrix[:, permutation_idx])
#     matched_box_l1_matrix = np.einsum('iij...->i...', box_l1_matrix[:, permutation_idx])

#     permuted_masks = masks[permutation_idx]

#     gt_points = gt_points[permutation_idx]

#     # points_in_masks, points_l1 = is_points_in_mask(permuted_masks, pred_points, gt_points)
#     # points_in_masks = np.sum(points_in_masks, axis=-1)

#     consistency, points_in_masks, points_l1 = None, None, None

#     return reward_assign(
#         object_number_match=object_number_match,
#         json_loadable=json_loadable,
#         valid_format_array=valid_format_array,
#         format_reward=format_reward,
#         consistency=consistency,
#         points_in_mask=points_in_masks,
#         points_l1=points_l1,
#         box_iou=matched_box_iou_matrix,
#         box_l1=matched_box_l1_matrix,
#     )


# def video_seg_reward(completion: str, **kwargs):
#     """For each frame, the model should output a grounding mask and a bounding box, in the following json format:
#     {
#         "0": {
#             "bbox": [x1, y1, x2, y2],
#             "points_1": [x, y],
#             "points_2": [x, y],
#         },
#         "1": {
#             "bbox": [x1, y1, x2, y2],
#             "points_1": [x, y],
#             "points_2": [x, y],
#         },
#         ...
#     }
#     the number is the index of the frame input to the model, corresponding to the order of masks in the input.
#     Args:
#         completions (str): the answer str of the model
#         kwarg (dict): The exactly same as the __getitem__
#     """
#     def is_valid_format(json_data, frame_number: int):
#         """检查每一帧输出的格式是否正确"""
#         frame_valid = np.zeros((frame_number,))

#         for frame_idx in json_data.keys():
#             try:
#                 # frame is mentioned, a mentioned-reward is assigned.
#                 frame_valid[int(frame_idx)] = frame_valid[int(frame_idx)] + FOUND_REWARD

#                 frame_data = json_data[frame_idx]
#                 if 'bbox_2d' not in frame_data: # or 'points_1' not in frame_data or 'points_2' not in frame_data:
#                     continue

#                 frame_idx = int(frame_idx)
#                 # 检查值的格式
#                 bbox = frame_data['bbox_2d']
#                 # points_1 = frame_data['points_1']
#                 # points_2 = frame_data['points_2']

#                 if isinstance(bbox, list) and len(bbox) == 4:
#                     frame_valid[frame_idx] = frame_valid[frame_idx] + BBOX_FORMAT_REWARD
#                 # if (isinstance(points_1, list) and len(points_1) == 2)\
#                 #     and (isinstance(points_2, list) and len(points_2) == 2) :
#                 #     frame_valid[frame_idx] = frame_valid[frame_idx] + POINTS_FORMAT_REWARD
#             except Exception:
#                 if frame_idx >= frame_number:
#                     frame_valid = frame_valid - FOUND_REWARD

#         return frame_valid


#     obj_num = kwargs.get('obj_number', None)
#     assert obj_num is not None and obj_num > 0, "obj_number should be greater than 0"

#     bboxs = kwargs.get('bboxs', None)
#     points = kwargs.get('points', None)
#     if bboxs is None or points is None:
#         raise ValueError("bboxs or points is None")

#     masks = kwargs.get('masks', None)
#     if masks.shape[0] != obj_num:
#         raise ValueError("masks group number mismatch with gt detected objects")

#     assert bboxs.shape[0] == points.shape[0] == masks.shape[0] == obj_num, "each object possess one set of bbox and points"
#     frame_number = len(kwargs.get('frames', None))

#     reward = 0.0
#     pred_datas = None
#     # if detected objects number consistent with gt objects number
#     obj_match = False
#     # if all the json str are valid
#     json_loadable = False
#     valid_format_array = np.zeros((obj_num, frame_number)).astype(np.uint8)
#     format_reward_array = np.zeros((obj_num, frame_number)).astype(np.int32)
#     # valid_frame_idx = None
#     # box_iou_reward = None
#     # box_l1_reward = None
#     # consistency = None
#     # points_in_mask = None
#     # points_l1_reward = None

#     try:
#         # 查找JSON对象
#         json_pattern = r'({[\s\S]*})'
#         json_match_list = re.findall(json_pattern, completion)

#         if len(json_match_list) == 0:
#             # 如果没有找到JSON对象，直接返回
#             return reward, None

#         if len(json_match_list) == obj_num:
#             obj_match = True

#         # json_str = json_match.group(1)
#         pred_datas = [json.loads(json_str) for json_str in json_match_list]
#         json_loadable = True

#         ############################
#         # if len(pred_datas) == 1:
#         #     pred_datas = [v for k, v in pred_datas[0].items()]

#         ############################


#         for obj, pred_data in enumerate(pred_datas):

#             format_reward = is_valid_format(pred_data, frame_number)
#             format_reward_array[obj] = format_reward

#             valid_frame_mask = format_reward == (FOUND_REWARD + BBOX_FORMAT_REWARD) # + POINTS_FORMAT_REWARD
#             # if valid_frame_mask.all():
#             valid_format_array[obj] = valid_frame_mask

#             # valid_frame_idx = np.where(valid_frame_mask).tolist()


#             # box_iou_reward = np.zeros_like(format_reward)
#             # box_l1_reward = np.zeros_like(format_reward)
#             # consistency = np.zeros_like(format_reward)
#             # points_in_mask = np.zeros_like(format_reward)
#             # points_l1_reward = np.zeros_like(format_reward)


#             # # 对有效的每一帧计算奖励
#             # for frame_idx in valid_frame_idx:
#             #     frame_data = pred_data[str(frame_idx)]
#             #     try:
#             #         frame_mask = masks[frame_idx]
#             #         # 生成真实边界框和关键点
#             #         gt_bbox = bboxs[frame_idx]
#             #         gt_points = points[frame_idx]


#             #         # 获取预测的边界框和关键点
#             #         pred_bbox = frame_data['bbox']
#             #         pred_points = [frame_data['points_1'], frame_data['points_2']]

#             #         # 计算边界框IoU, 后续计算奖励
#             #         box_iou_reward[frame_idx] = box_iou(pred_bbox, gt_bbox)

#             #         # 计算边界框L1距离
#             #         box_l1_reward[frame_idx] = box_l1_distance(pred_bbox, gt_bbox)

#             #         consistency[frame_idx]= (points_in_box(pred_points[0], pred_bbox)
#             #                                 and
#             #                                 points_in_box(pred_points[1], pred_bbox))

#             #         points_in_mask[frame_idx] = is_points_in_mask(frame_mask, pred_points)

#             #         points_l1_reward[frame_idx] = points_distance(pred_points, gt_points)

#                 # except Exception as e:
#                 #     continue


#     except Exception as e:
#         # 处理JSON解析错误(normally this won't raise)
#         return reward, None

#     video_resized_hw = kwargs['preprocess_kwargs'].get('video_resized_hw', None)[0]
#     video_ori_hw = kwargs['preprocess_kwargs'].get('video_ori_hw', None)[0]
#     if video_resized_hw is None or video_ori_hw is None:
#         raise ValueError("video_resized_hw or video_ori_hw is None")

#     if isinstance(masks, list):
#         masks = np.stack(masks, axis=0)
#     assert masks.ndim == 4, "masks should be 4D np.array of (obj_num, frame_num, H, W)"

#     masks = torch.from_numpy(masks).float()

#     scaled_masks = torch.nn.functional.interpolate(
#         masks, size=video_resized_hw, mode='nearest'
#     ).int().cpu().numpy()

#     h_ratio = video_resized_hw[0] / float(video_ori_hw[0])
#     w_ratio = video_resized_hw[1] / float(video_ori_hw[1])

#     # bboxs = np.stack(bboxs, axis=0)
#     # points = np.stack(points, axis=0)
#     scale = np.array([w_ratio, h_ratio, w_ratio, h_ratio]).astype(np.float32)
#     assert bboxs.ndim == 3 and points.ndim == 3, "bboxs and points should be 3D np.array of (obj_num, frame_num, 4)"

#     scaled_bboxs = bboxs.astype(np.float32) * scale
#     scaled_points = points.astype(np.float32) * scale

#     reward, reward_dict = calculate_reward(
#         object_number_match = obj_match,
#         json_loadable=json_loadable,
#         valid_format_array = valid_format_array,
#         # valid_frame_mask = valid_frame_mask,
#         format_reward=format_reward_array,
#         obj_number=obj_num,
#         masks = scaled_masks,
#         gt_bboxs = scaled_bboxs.astype(np.int32),
#         gt_points = scaled_points.astype(np.int32),
#         pred_datas = pred_datas
#         # box_iou=box_iou_reward,
#         # box_l1=box_l1_reward,
#         # consistency=consistency,
#         # points_in_mask=points_in_mask,
#         # points_l1=points_l1_reward
#     )
#     return reward, reward_dict
#     # if valid_frame_idx.sum() != format_reward.shape[0]:
#     #         # if not all frames are format-correct, return format reward only
#     #         return reward

# 定义奖励权重
# MASK_EXISTED_REWARD = 0.3  # 时间间隔内有mask的奖励
MASK_RATIO_REWARD = 1.0  # mask覆盖率奖励
INTERVAL_LENGTH_REWARD = 0.2  # 时间间隔长度奖励

MATCH_REWARD = 0.2  # 匹配奖励
ELEMENT_REWARD = 0.2  # start end description 都出现的奖励
TIME_CONSISTENCY_REWARD = 0.3  # 时间一致性奖励
OVER_LAP_PENELTY = 0.1  # 重叠惩罚

# 帧率参数，FPS=2，表示每秒两帧
FPS = 2


# def time_reward(completion: str, **kwargs):
#     """评估视频时间间隔标记任务的奖励函数

#     奖励条件:
#     1. 回答的时间间隔内出现过mask，有奖励
#     2. 间隔内出现mask的帧数比例越高，分数越高
#     3. 时间间隔不能太长，必须在几张图片的数量之内，最好是4张以内

#     Args:
#         completion (str): 模型输出的答案字符串
#         **kwargs: 附加参数，包含masks等信息

#     Returns:
#         float: 奖励值，范围[0, 1]
#     """

#     cumulated_reward = 0.

#     try:
#         # 提取JSON对象

#         json_pattern = r'({[\s\S]*?})'
#         json_match = re.search(json_pattern, completion)

#         if not json_match:
#             return cumulated_reward, None

#         cumulated_reward += MATCH_REWARD

#         json_str = json_match.group(1)
#         result = json.loads(json_str)

#         # 提取时间戳
#         start_time = result.get('start_time', None)
#         end_time = result.get('end_time', None)
#         description = result.get('description', None)

#         # 验证时间戳格式
#         time_pattern = r'(\d+):(\d+)\.(\d+)'
#         start_match = re.match(time_pattern, start_time)
#         end_match = re.match(time_pattern, end_time)

#         if not (start_match is not None and end_match is not None and description is not None):
#             return cumulated_reward, None

#         cumulated_reward += ELEMENT_REWARD

#         # 解析时间戳为秒数
#         def parse_time_to_seconds(time_str):
#             minutes, seconds, ms = map(int, re.match(time_pattern, time_str).groups())
#             return minutes * 60 + seconds + ms*10 / 1000

#         start_seconds = parse_time_to_seconds(start_time)
#         end_seconds = parse_time_to_seconds(end_time)

#         # 计算开始和结束的帧索引
#         frame_count = len(kwargs.get('frames', []))
#         total_duration = frame_count / FPS

#         if start_seconds >= end_seconds:
#             return cumulated_reward, None

#         if end_seconds > total_duration:
#             return cumulated_reward, None

#         cumulated_reward += TIME_CONSISTENCY_REWARD

#         # 将时间转为帧索引
#         start_frame_idx = min(int(start_seconds * FPS), frame_count - 1)
#         end_frame_idx = min(int(end_seconds * FPS), frame_count - 1)

#         # 计算时间间隔的帧数
#         interval_frames = end_frame_idx - start_frame_idx + 1

#         # 获取masks信息
#         masks = kwargs.get('masks')
#         if masks is None:
#             raise ValueError("masks is None")

#         # 计算时间间隔内每一帧是否有mask
#         masks_sum = np.sum(masks, axis=(2, 3))  # 对所有对象、高度和宽度求和

#         # FIXME: 这里假设masks的形状为 [obj_num, frame_num, H, W], obj_num == 1
#         selected_masks = masks_sum[:, start_frame_idx:end_frame_idx + 1]

#         # 计算有多少帧包含mask (非零值)
#         frames_with_mask = np.count_nonzero(selected_masks > 0)

#         # 条件1: 时间间隔内有mask
#         mask_existed = frames_with_mask > 0
#         mask_existed_reward = MASK_EXISTED_REWARD if mask_existed else 0.0
#         cumulated_reward += mask_existed_reward

#         # 条件2: 时间间隔内有mask的帧数比例
#         mask_ratio = (frames_with_mask / (interval_frames*selected_masks.shape[0])) if interval_frames > 0 else 0
#         mask_ratio_reward = MASK_RATIO_REWARD * mask_ratio
#         cumulated_reward += mask_ratio_reward

#         # 条件3: 时间间隔长度奖励 (最佳为4帧或以下，超过4帧奖励降低)
#         # 条件3: 时间间隔长度奖励 (最佳为4帧或以下，超过4帧奖励降低)
#             # if interval_frames <= 2:
#             #     interval_length_reward = INTERVAL_LENGTH_REWARD * (1.5 if interval_frames == 1 else 1.0)
#             # else:
#             #     # 衰减系数，每增加1帧降低25%的奖励
#             #     decay = max(0, 1 - 0.5 * (interval_frames - 2))
#             #     interval_length_reward = INTERVAL_LENGTH_REWARD * decay

#         length_reward_o_pennelty = (2 - interval_frames) * INTERVAL_LENGTH_REWARD

#         cumulated_reward += length_reward_o_pennelty
#         grounding_indices = np.linspace(start_frame_idx, end_frame_idx, end_frame_idx - start_frame_idx + 1, dtype=int).tolist()
#         descriptions = [description] * (end_frame_idx - start_frame_idx + 1)

#         reward_dict = {"cumulated_reward": cumulated_reward,
#                        "mask_ratio_reward": mask_ratio_reward,
#                        "interval_length_reward": length_reward_o_pennelty,
#                        "grounding": {"grounding_indices": grounding_indices, "descriptions":descriptions, "valid_mask": selected_masks > 0},
#                        "kwargs": kwargs}

#         return cumulated_reward, reward_dict

#         # iou_reward = grounding_process_reward(start_frame_idx, end_frame_idx, description, selected_masks > 0, **kwargs)

#         # # 总奖励
#         # total_reward = mask_existed_reward + mask_ratio_reward + interval_length_reward + iou_reward
#         # cumulated_reward += total_reward


#         # return cumulated_reward, reward_dict

#     except Exception as e:
#         print(f"Error in timestamp_reward: {e}")
#         return cumulated_reward, None


def make_prompt(frames: List[str], descriptions: List[str]) -> dict:
    """生成用于Grounding的提示词

    Args:
        frames (List[str]): 输入的帧路径列表
        description (str): 描述文本

    Returns:
        dict: 包含提示词的字典
    """
    # img_num = len(frames)
    inputs = []

    for i, (frame, description) in enumerate(zip(frames, descriptions)):
        # text = ORIGINAL_GROUNDING_TEMPLATE.format(
        #     Question = descriptions.lower().strip(),
        #     Answer = origin_grounding_answer
        # ) if do_thinking else NOTHINKING_GROUNDING_TEMPLATE.format(
        #     Question = description.lower().strip(),
        #     Answer = origin_grounding_answer
        # )
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


import time

from PIL import ImageDraw


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

    pass


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

    ious = []
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
    valid_l1s = l1_matrix_frame[valid_mask]

    selected_masks = (masks[:, array_indices, :] > 0).astype(np.int32)
    masks = (masks > 0).astype(np.int32)
    # gt_selected_mask_sum = np.sum(selected_masks, axis=(2, 3)) / np.prod(selected_masks.shape[-2:])  # 对所有对象、高度和宽度求和
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
    is_multi_frame_and_multi_bbox = pred_frame_valid_indices.shape[0] > 1 and np.any(
        frame_counts > 1
    )

    avg_iou = np.mean(valid_ious) if valid_ious.size > 0 else 0.0
    # avg_ratio = np.mean(gt_selected_mask_ratio) if gt_selected_mask_ratio.size > 0 else 0.0
    avg_ratio = (
        np.mean(gt_selected_mask_ratio) if gt_selected_mask_ratio.size > 0 else 0.0
    )

    use_sam = os.getenv("USE_SAM")
    offset = float(offset / (float(np.prod(bboxs.shape[:2])) + 1e-6))

    if use_sam is None or use_sam != "true":
        print(f"avg_iou: {avg_iou}，avg_ratio:{avg_ratio}")
        return {
            "object_num_mismatch": offset,
            "avg_iou": float(avg_iou),
            "avg_ratio": float(avg_ratio),
        }

    if use_sam is not None and use_sam == "true":
        if avg_iou < 0.3 and os.getenv("USE_SAM_REWARD_ONLY") != "true":
            print(f"avg_iou: {avg_iou}，回退到boxs IoU Avg")

            if os.getenv("PLOG") == "true":
                write_log(
                    idx,
                    raw_problem,
                    inputs,
                    completions,
                    arg_dict,
                    valid_mask,
                    iou_matrix_frame,
                    pred_bbox,
                    scaled_bboxs,
                )

            return {
                "object_num_mismatch": offset,
                "avg_iou": float(avg_iou),
                "avg_ratio": float(avg_ratio),
            }

        else:
            if os.getenv("USE_SAM_REWARD_ONLY") == "true":
                print("仅使用SAM2分割分数奖励")

            print(f"avg_iou: {avg_iou}，使用SAM2视频分数奖励")
            img_list = kwargs.get("frames", None)
            img_list = [pth(img).resolve() for img in img_list]
            video_dir = str(img_list[0].parent)
            frame_names = [
                img.name for img in img_list
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

            # [ [obj_id, frame_id, np.ndarray(4)], [obj_id, frame_id, np.ndarray(4)], ... ]

            # for obj in range(pred_bbox.shape[0]):
            #     permuted_pred_bboxs_dict[obj] = {}
            #     for frame in range(pred_bbox.shape[1]):
            #         # 只添加有效的预测边界框
            #         if np.all(permuted_pred_bboxs[obj, frame, :] > -1):
            #             permuted_pred_bboxs_dict[obj][frame] = np.array(permuted_pred_bboxs[obj, frame, :])

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


import cv2


def sam2_single_reward(
    response: Union[Dict | List],
    gt_masks: np.array,
    images: list,
    avg_iou: float,
    avg_ratio: float,
    object_num_mismatch: float,
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
    print(
        f"avg_iou:{avg_iou}, avg_ratio:{avg_ratio}, offset: {object_num_mismatch}, overall sam metric:{final_metric}"
    )
    if os.getenv("USE_SAM_REWARD_ONLY") == "true":
        # print("############ 仅使用SAM2分割分数奖励,确保所有展开仅使用。 ############")
        return (
            float(1.0 + 2 * final_metric) + object_num_mismatch
        )  # to avoid the avg_ratio perturbation

    return float(1.0 + 2 * final_metric + avg_iou + avg_ratio + object_num_mismatch)


def sam2_gather_rewards(responses: list):
    """根据SAM2的响应计算奖励"""
    rewards = [sam2_single_reward(**response) for response in responses]
    print(f"RANK[{torch.distributed.get_rank()}] SAM2 分割分数: {rewards}")
    return rewards


def cal_metric(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray]):
    """_summary_

    Args:
        pred_masks (List[np.ndarray]): shaped as (n_frames, H, W)
        gt_masks (List[np.ndarray]): shaped as (n_frames, H, W)
    """
    j = db_eval_iou(gt_masks, pred_masks).mean()
    return j
    f = db_eval_boundary(gt_masks, pred_masks).mean()
    a = get_r2vos_accuracy(gt_masks, pred_masks).mean()
    # FIXME: HAS NO foreground_masks
    # r = get_r2vos_robustness(gt_masks, pred_masks, foreground_masks).mean()


# def grounding_process_reward(gprompts: List[List[str]], gcompletions: List[List[str]], gdict_list: List[List[str]], kwargs_list, key_dicts, temp_rewards):

#     rewards = []
#     # 根据返回的结果处理每个 gprompt 对应的奖励
#     for idx, (args, key_dict, temp_reward) in enumerate(zip(kwargs_list, key_dicts, temp_rewards)):
#         if key_dict is None: # in dummy_idx:
#             # 如果是 dummy prompt，直接使用 temp_reward
#             rewards.append(temp_reward)

#         if key_dict is not None:
#             iou_reward = grounding_process_reward(
#                 idx=idx,
#                 raw_problem=args['problem'],
#                 inputs=gprompts[idx],
#                 grounding_indices=key_dict['grounding']['grounding_indices'],
#                 valid_mask=key_dict['grounding']['valid_mask'],
#                 completions=gcompletions[idx],
#                 arg_dict=gdict_list[idx],
#                 **key_dict['kwargs']
#             )
#             key_dict['iou_reward'] = iou_reward
#             key_dict['cumulated_reward'] = key_dict['cumulated_reward'] + iou_reward
#             key_dict['total_reward'] = key_dict['cumulated_reward']

#             # 删除不必要的键
#             del key_dict['grounding']
#             del key_dict['kwargs']

#             rewards.append(key_dict['total_reward'])


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

    if len(json_match) == 0:
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

            if start_seconds >= end_seconds or end_seconds > total_duration:
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

        # 条件3: 时间间隔长度奖励 (最佳为4帧或以下，超过4帧奖励降低)
        # if interval_frames <= 2:
        #     interval_length_reward = INTERVAL_LENGTH_REWARD * (1.5 if interval_frames == 1 else 1.0)
        # else:
        #     # 衰减系数，每增加1帧降低25%的奖励
        #     decay = max(0, 1 - 0.5 * (interval_frames - 2))
        #     interval_length_reward = INTERVAL_LENGTH_REWARD * decay

        length_reward_o_penalty = (2 - interval_frames) * INTERVAL_LENGTH_REWARD
        length_reward_o_penalty = (
            length_reward_o_penalty if length_reward_o_penalty < 0 else 0
        )

        ratio_rewards.append(mask_ratio_reward)
        length_rewards.append(length_reward_o_penalty)

    all_indices = np.concatenate(valid_frame_idx)
    over_lap = len(all_indices) - len(np.unique(all_indices))
    over_lap_penelty = over_lap * OVER_LAP_PENELTY

    cumulated_reward += sum(ratio_rewards) + sum(length_rewards) - over_lap_penelty
    cumulated_reward = cumulated_reward / js_number

    N = 5
    all_g_indices = np.unique(all_indices)
    if all_g_indices.shape[0] > N:
        cumulated_reward = (
            cumulated_reward - (len(all_g_indices) - N) * 2 * INTERVAL_LENGTH_REWARD
        )

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
        "grounding": {
            "grounding_indices": g_indices,
            "descriptions": description_list,
            "valid_mask": selected_masks > 0,
        },
        "kwargs": kwargs,
    }

    return cumulated_reward, reward_dict


# def grounding2_reward(idx:int, raw_problem: str, inputs: list, grounding_indices: list, valid_mask: np.array, completions: list, arg_dict: list, **kwargs):

#     masks = kwargs.get('masks', None)
#     bboxs = kwargs.get('bboxs', None)
#     points = kwargs.get('points', None)

#     assert masks is not None and bboxs is not None and points is not None, \
#         f"data items {masks if masks is None else bboxs if bboxs is None else points} should not be None"

#     masks = torch.from_numpy(masks).float()
#     assert bboxs.ndim == 3 and points.ndim == 3, "bboxs and points should be 3D np.array of (obj_num, frame_num, 4)"


#     pred_bbox = [[] for _ in range(bboxs.shape[0])]
#     scaled_bboxs = np.array(bboxs).astype(np.float32)
#     offset = 0.

#     for i, (completion, data_info) in enumerate(zip(completions, arg_dict)):
#         # completion = completion.strip()
#         # print(f"completion: {completion}")

#         img_resized_hw = data_info.get('img_resized_hw', None)[0]
#         img_ori_hw = data_info.get('img_ori_hw', None)[0]
#         if img_resized_hw is None or img_ori_hw is None:
#             raise ValueError("img_resized_hw or img_ori_hw is None")

#         h_ratio = img_resized_hw[0] / float(img_ori_hw[0])
#         w_ratio = img_resized_hw[1] / float(img_ori_hw[1])

#         # bboxs = np.stack(bboxs, axis=0)
#         # points = np.stack(points, axis=0)
#         scale = np.array([w_ratio, h_ratio, w_ratio, h_ratio]).astype(np.float32)

#         # scaled_masks = torch.nn.functional.interpolate(
#         #     masks, size=img_resized_hw, mode='nearest'
#         # ).int().cpu().numpy()
#         c_idx = grounding_indices[i]
#         scaled_bboxs[:, c_idx, :] = bboxs[:, c_idx, :].astype(np.float32) * scale
#         # scaled_points = points.astype(np.float32) * scale


#         # ans = extract_answer(completion)
#         ans = completion
#         json_pattern = r'({[\s\S]*?})'
#         js_match_list = re.findall(json_pattern, ans)

#         cnt = 0
#         for js in js_match_list:
#             if cnt == len(pred_bbox):
#                 offset = offset - (len(js_match_list) - bboxs.shape[0]) * OUT_OF_GT
#                 break
#             try:
#                 js_result = json.loads(js)
#                 assert isinstance(js_result['bbox_2d'], list) and len(js_result['bbox_2d']) == 4, "bbox_2d should have 4 elements"
#                 pred_bbox[cnt].append(js_result['bbox_2d'])
#                 cnt += 1
#             except Exception as e:
#                 pred_bbox[cnt].append([-1, -1, -1, -1])
#                 cnt += 1
#         if cnt < bboxs.shape[0]:
#             for j in range(cnt, bboxs.shape[0]):
#                 pred_bbox[j].append([-1, -1, -1, -1])

#         # pred_bboxes: np.ndarray [obj_num, frame_num, 4]

#     array_indices = np.array(grounding_indices).astype(np.int32)

#     try:
#         pred_bbox = np.array(pred_bbox).astype(np.float32)
#         scaled_bboxs = scaled_bboxs[:, array_indices, :]
#         # ious = iou(pred_bbox, scaled_bboxs)
#         iou_matrix_frame, l1_matrix_frame = iou_framewise_object_matching(scaled_bboxs, pred_bbox)

#         valid_ious = iou_matrix_frame[valid_mask]
#         valid_l1s = l1_matrix_frame[valid_mask]


#     except Exception as e:
#         print(f"Error in grounding_process_reward: {e}")
#         # print(f'{valid_indices.shape}, {ious.shape}')
#         return 0.0 + offset

#     if os.getenv('PLOG') == 'true':
#         write_log(idx, raw_problem, inputs, completions, arg_dict, valid_mask, iou_matrix_frame, pred_bbox, scaled_bboxs)

#     return offset + (valid_ious.sum() / (np.prod(valid_ious.shape) + 1e-4))


def seg_bidirectional_from_bbox(
    predictor,
    video_dir,
    frame_names,
    pred_bbox,
    ann_frame_idx,
    cache_dir="./.seg_cache",
):
    """
    使用 diskcache 进行正向 + 反向分割推理。支持多目标 + 多关键帧，每个目标合并推理一次。

    参数:
        predictor: SAM2预测器
        video_dir: 视频帧目录
        frame_names: 所有帧名列表
        pred_bbox: np.ndarray, [obj_num, frame_num, 4]
        ann_frame_idx: List[int]，对应 pred_bbox 中每帧 bbox 的实际帧编号
        cache_dir: 缓存路径（持久）

    返回:
        video_segments: dict[int, dict[int, np.ndarray]]
    """
    pred_bbox = np.asarray(pred_bbox)
    obj_num, frame_num, _ = pred_bbox.shape
    assert len(ann_frame_idx) == frame_num, "ann_frame_idx 应与 pred_bbox 第二维一致"

    total_frames = len(frame_names)
    print(total_frames, obj_num, frame_num)
    video_segments = {}

    if os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    with dc.Cache(cache_dir) as cache:
        base_path = os.path.join(cache_dir, "video_seg_cache")
        os.makedirs(base_path, exist_ok=True)

        # === 正向推理 ===
        forward_dir = os.path.join(base_path, "forward_video")
        os.makedirs(forward_dir, exist_ok=True)

        # 拷贝所有帧，保持原顺序
        for i, fname in enumerate(frame_names):
            dst = os.path.join(forward_dir, f"{i:05d}.jpg")
            if not os.path.exists(dst):
                shutil.copy(os.path.join(video_dir, fname), dst)

        inference_state = predictor.init_state(video_path=forward_dir)

        for obj_id in range(obj_num):
            for i, frame_idx in enumerate(ann_frame_idx):
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=pred_bbox[obj_id, i],
                )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            for i, out_obj_id in enumerate(out_obj_ids):
                if out_frame_idx not in video_segments:
                    video_segments[out_frame_idx] = {}
                video_segments[out_frame_idx][out_obj_id] = (
                    (out_mask_logits[i] > 0).cpu().numpy()
                )

        # === 反向推理 ===
        reverse_dir = os.path.join(base_path, "reverse_video")
        os.makedirs(reverse_dir, exist_ok=True)

        # 拷贝帧：倒序编号
        for i in range(total_frames):
            src = os.path.join(video_dir, frame_names[i])
            dst = os.path.join(reverse_dir, f"{total_frames - 1 - i:05d}.jpg")
            if not os.path.exists(dst):
                shutil.copy(src, dst)

        inference_state = predictor.init_state(video_path=reverse_dir)

        for obj_id in range(obj_num):
            for i, original_idx in enumerate(ann_frame_idx):
                rev_idx = total_frames - 1 - original_idx  # 在倒序视频中的索引
                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=rev_idx,
                    obj_id=obj_id,
                    box=pred_bbox[obj_id, i],
                )

        for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
            inference_state
        ):
            # 反映射回来
            real_idx = total_frames - 1 - out_frame_idx
            for i, out_obj_id in enumerate(out_obj_ids):
                if real_idx not in video_segments:
                    video_segments[real_idx] = {}
                if out_obj_id not in video_segments[real_idx]:
                    video_segments[real_idx][out_obj_id] = (
                        (out_mask_logits[i] > 0).cpu().numpy()
                    )

    return video_segments
