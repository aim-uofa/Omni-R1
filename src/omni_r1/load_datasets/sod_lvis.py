import json
import os
import random
import re
from pathlib import Path as pth
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from torchvision.ops import box_iou

VISUALIZATION_PATH = pth("./visualizations")
from PIL import Image

QUESTION_TEMPLATE = (
    "Find up to three different regions in the image that likely contain a high number of '{object}'. "
    "Even if the '{object}' are not clearly visible, infer where they are most likely to appear. "
    "Each region should cover multiple '{object}' and include some visual context. "
    "The selected regions should be as distinct as possible, with minimal or no overlap between them. "
    "Return the coordinates in JSON format as: "
    '{"bbox_2d": [x1, y1, x2, y2], "label": "{object}-dense region"}. '
    "Explain your reasoning in <think>...</think> and output the final result in <answer>...</answer>."
    "i.e., <think> thinking process here </think>"
    "<answer> JSON format here </answer>"
)

answer = "{ 'bbox' : [10,100,200,210], 'points_1' : [30,110], 'points_2' : [35,180] }"
MAX_DATASET_SIZE = 1000


class SOD_LVISDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 32,
        num_proc: int = 16,
        *args,
        **kwargs,
    ):
        dataset_path = pth(dataset_path).resolve()
        if not dataset_path.exists():
            raise ValueError(f"Seg-Zero Dataset path {dataset_path} does not exist.")
        self.dataset_path = str(dataset_path)
        assert self.dataset_path.endswith(".json"), (
            "SOD_LVIS dataset path should be a json file"
        )
        with open(self.dataset_path, "r") as f:
            self.raw_dataset = json.load(f)
        self.raw_dataset = self.raw_dataset[:MAX_DATASET_SIZE]
        # get data dir
        self.data_dir = os.path.dirname(self.dataset_path)
        self.dataset_name = "coco"
        self.data_dir = self.data_dir + "/coco"
        # self.rank0_load(batched= batch_size > 1, batch_size=batch_size, num_proc=num_proc)

        # self.dataset = load_dataset(self.dataset_path)
        # self.load_data(batched= batch_size > 1, batch_size=batch_size, num_proc=num_proc)

    def rank0_load(self, **kwargs):
        # 获取分布式环境信息
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        cache_path = "./processed_dataset_cache"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # 仅在主进程执行预处理,其他进程等待
        if local_rank == 0:
            self.dataset = load_dataset(self.dataset_path)
            self.load_data(**kwargs)
            # 保存到共享位置
            # cache_path = "./tmp/processed_dataset_cache"
            self.dataset.save_to_disk(cache_path)

        # 同步所有进程
        if world_size > 1:
            torch.distributed.barrier()

        # 所有进程从缓存加载
        if local_rank != 0:
            self.dataset = load_from_disk(cache_path)

    def load_data(self, **kwargs):
        def make_conversation_image_and_video(examples):
            batch_size = len(examples["problem"])

            # 初始化要返回的批处理结果
            results = {
                "data_type": ["image"] * batch_size,
                "problem_type": ["image-segmentation"] * batch_size,
                "problem_id": examples["id"],
                "prompt": [],
            }

            # 处理每个样本
            for i in range(batch_size):
                question = examples["problem"][i]
                prompt = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": examples["image"][i],
                                "resized_height": examples["img_height"][i],
                                "resized_width": examples["img_width"][i],
                            },
                            {
                                "type": "text",
                                "text": QUESTION_TEMPLATE.format(
                                    Question=question.lower().strip("."), Answer=answer
                                ),
                            },
                        ],
                    }
                ]

                results["prompt"].append(prompt)

            return results

        self.dataset = (
            self.dataset["train"]
            .select(range(10))
            .map(
                make_conversation_image_and_video,
                remove_columns=["id", "img_width", "img_height"],
                **kwargs,
            )
        )

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        idx = [idx]
        items_list = []
        for i in idx:
            items = self.raw_dataset[i]
            items_list.append(items)
        items = {k: [d[k] for d in items_list] for k in items_list[0].keys()}
        bs = len(idx)
        items["prompt"] = []
        items["data_type"] = ["image"] * bs
        items["problem_type"] = ["sod"] * bs
        # items['problem_id'] = items['id']
        items["annotation"] = []
        items["object"] = []
        items["image"] = []
        items["problem_id"] = []

        for i in range(bs):
            file_name = items["file_name"][i]
            # items['image'][i] = pth(self.dataset_path).parent / file_name
            dense_object_set = set(items["dense_object_list"][i])
            small_object_set = set(items["small_object_list"][i])
            # random select a object
            obtect_set = dense_object_set.union(small_object_set)
            assert len(obtect_set) > 0, "object set is empty"
            object = random.choice(list(obtect_set))
            question = object  # for sod,
            ori_height = items["height"][i]
            ori_width = items["width"][i]
            # resize short side to 1024
            short_side = min(ori_height, ori_width)
            scale = 1024 / short_side
            target_height = int(ori_height * scale)
            target_width = int(ori_width * scale)

            prompt = [
                {
                    "role": "user",
                    "content": [
                        # {
                        #     "type": "image",
                        #     "image": os.path.join(self.data_dir, file_name),
                        #     #"min_pixels": 1024*28*28,
                        #     "resized_height": target_height,
                        #     "resized_width": target_width,
                        # },
                        {
                            "type": "text",
                            "text": QUESTION_TEMPLATE.replace("{object}", object),
                        },
                        {
                            "type": "image",
                            "image": os.path.join(self.data_dir, file_name),
                            # "min_pixels": 1024*28*28,
                            "resized_height": target_height,
                            "resized_width": target_width,
                        },
                    ],
                }
            ]
            annotation = items["categories2ann"][i][object]
            items["prompt"].append(prompt)
            items["annotation"].append(annotation)
            items["object"].append(object)
            items["image"].append(os.path.join(self.data_dir, file_name))
            image_id = int(file_name.split(".")[0].split("/")[-1])
            items["problem_id"].append(f"{self.dataset_name}_{image_id}")

        f_items = {k: v[0] for k, v in items.items()}
        return f_items

    # {'id': ['refcocog_16521'], 'problem': ['A black and white dog laying down, looking away from the camera.'], 'solution': ['<box>(0,457),(374,672)</box><points>(50,592),(144,601)</points>'], 'image': [<PIL.PngImagePlugin.PngImageFile image mode=RGB size=840x840 at 0x7F629EF21490>], 'img_height': [426], 'img_width': [640], 'prompt': [[...]], 'data_type': ['image'], 'problem_type': ['image-segmentation'], 'problem_id': ['refcocog_16521']}


def draw_example(
    resized_hw, image, problem: str, gt_box_list, pred_box_list, problem_id: str
):
    """
    Visualize ground truth and predicted boxes on the same image.

    Args:
        resized_hw (tuple): (resized_height, resized_width)
        image (PIL.Image): original image
        problem (str): text description
        gt_box_list (list): list of GT boxes [[x1,y1,x2,y2], ...]
        pred_box_list (list): list of predicted boxes [[x1,y1,x2,y2], ...]
        problem_id (str): unique ID for saving
    """
    if not isinstance(image, Image.Image):
        assert isinstance(image, str), "image should be a PIL image or a path"
        image = Image.open(image).convert("RGB")

    w, h = image.size
    rh, rw = resized_hw

    # Resize image
    resized_image = image.resize((rw, rh))
    resized_image_cv = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)

    # Draw GT boxes (Green)
    if gt_box_list is not None and len(gt_box_list) > 0:
        for box in gt_box_list:
            x1, y1, x2, y2 = box
            cv2.rectangle(
                resized_image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
            )

    # Draw Predicted boxes (Blue)
    if pred_box_list is not None and len(pred_box_list) > 0:
        for box in pred_box_list:
            x1, y1, x2, y2 = box
            cv2.rectangle(
                resized_image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2
            )

    # Draw Problem text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resized_image_cv, problem, (10, 30), font, 0.7, (255, 0, 0), 2)

    # Save image
    output_dir = pth(VISUALIZATION_PATH).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{problem_id}"
    output_path.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(output_path / "combined.jpg"), resized_image_cv)

    return str(output_path)


def sod_reward(
    completion: str,
    gt: str,
    resized_hw: Tuple[float, float],
    strict: bool = False,
    **kwargs,
) -> Tuple[float, Optional[str], Optional[str | pth]]:
    """compute rewards unique to segmentation task, including seg_format, bbox_iou, bbox_l1 and point_l1.
    Only for 'video segmentation' question type, otherwise return 0.

    Args:
        completion (str): predicted completion string
        gt (str): ground truth string
        resized_hw (Tuple[float, float]): resized height and width
        strict (bool, optional): whether to use strict format checking. Defaults to False.
    """

    def convert_bbox_to_xyxy(box_list):
        """
        Convert bounding boxes from (x, y, w, h) to (x1, y1, x2, y2) format.
        Args:
            box_list (list or np.ndarray): (N, 4) list or array of boxes [x, y, w, h]
        Returns:
            np.ndarray: (N, 4) array of boxes [x1, y1, x2, y2]
        """
        boxes = np.array(box_list, dtype=np.float32)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2] - 1
        y2 = boxes[:, 1] + boxes[:, 3] - 1
        return np.stack((x1, y1, x2, y2), axis=1)

    def compute_iou(box_list1, box_list2):
        """
        Compute pairwise IoU between two lists of boxes.

        Args:
            box_list1 (list or np.ndarray): (N, 4) list or array of boxes [x1, y1, x2, y2]
            box_list2 (list or np.ndarray): (M, 4) list or array of boxes [x1, y1, x2, y2]

        Returns:
            np.ndarray: (N, M) IoU matrix
        """
        if len(box_list1) == 0 or len(box_list2) == 0:
            # 注意：这里返回空矩阵而不是0.0，更符合矩阵操作习惯
            return np.zeros((len(box_list1), len(box_list2)), dtype=np.float32)

        box1 = np.array(box_list1, dtype=np.float32)
        box2 = np.array(box_list2, dtype=np.float32)

        box1_tensor = torch.from_numpy(box1)
        box2_tensor = torch.from_numpy(box2)

        # Compute IoU using torchvision's optimized function
        iou = box_iou(box1_tensor, box2_tensor)

        return iou.cpu().numpy()

    def compute_intersection(gt_boxes, pred_boxes):
        """
        Compute pairwise intersection areas between gt_boxes and pred_boxes.

        Args:
            gt_boxes (Tensor or list): (N,4) ground truth boxes [x1, y1, x2, y2]
            pred_boxes (Tensor or list): (M,4) predicted boxes [x1, y1, x2, y2]

        Returns:
            Tensor: (N, M) matrix of intersection areas
        """
        # 自动转为Tensor
        if not isinstance(gt_boxes, torch.Tensor):
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        if not isinstance(pred_boxes, torch.Tensor):
            pred_boxes = torch.tensor(pred_boxes, dtype=torch.float32)

        assert gt_boxes.ndim == 2 and gt_boxes.shape[1] == 4, "gt_boxes must be (N,4)"
        assert pred_boxes.ndim == 2 and pred_boxes.shape[1] == 4, (
            "pred_boxes must be (M,4)"
        )

        # 广播处理 (N, 1, 4) 和 (1, M, 4)
        max_xy = torch.min(gt_boxes[:, None, 2:], pred_boxes[None, :, 2:])  # (N, M, 2)
        min_xy = torch.max(gt_boxes[:, None, :2], pred_boxes[None, :, :2])  # (N, M, 2)

        # 相交区域的宽和高
        inter_wh = (max_xy - min_xy + 1).clamp(min=0)  # 防止负数
        inter_area = inter_wh[:, :, 0] * inter_wh[:, :, 1]  # 宽 * 高

        return inter_area  # (N, M)

    def compute_coverage_rate(target_box_list, pred_box_list, min_coverage=0.95):
        """
        Count how many target boxes are sufficiently covered by predicted boxes based on coverage ratio.

        Args:
            target_box_list (list or np.ndarray): (N,4)
            pred_box_list (list or np.ndarray): (M,4)
            min_coverage (float): minimum coverage ratio to be considered covered (default=0.95)

        Returns:
            int: number of targets covered
            float: coverage ratio
        """
        if len(target_box_list) == 0 or len(pred_box_list) == 0:
            return 0, 0.0

        # 转Tensor
        target_boxes = torch.tensor(target_box_list, dtype=torch.float32)
        pred_boxes = torch.tensor(pred_box_list, dtype=torch.float32)

        target_area = (target_boxes[:, 2] - target_boxes[:, 0] + 1) * (
            target_boxes[:, 3] - target_boxes[:, 1] + 1
        )  # (N,)

        # 交集面积矩阵 (N, M)
        inter_area = compute_intersection(target_boxes, pred_boxes)

        # 每个gt，每个pred的覆盖率
        coverage_ratios = inter_area / (target_area[:, None] + 1e-6)  # (N,M)

        # 每个target取最大的覆盖率
        max_coverage = coverage_ratios.max(dim=1).values  # (N,)

        # 判断哪些target被足够覆盖
        covered = max_coverage >= min_coverage

        covered_num = covered.sum().item()
        coverage_ratio = covered_num / len(target_box_list)

        return covered_num, coverage_ratio

    def fast_no_overlap_reward(box_list, iou_threshold=0.3):
        if len(box_list) <= 1:
            return 1.0  # 0个或1个框肯定不会overlap

        # 计算IoU矩阵
        iou = compute_iou(box_list, box_list)
        # 对角线 元素为1，去掉
        np.fill_diagonal(iou, 0)

        # 只要有任何一对超过阈值，就判定为失败
        if np.any(iou > iou_threshold):
            return 0.0
        else:
            return 1.0

    def compute_areas_numpy(box_list):
        """
        Compute areas of boxes using numpy (batch).
        Args:
            box_list (list or np.ndarray): (N,4) list of boxes [x1, y1, x2, y2]
        Returns:
            np.ndarray: (N,) array of areas
        """
        boxes = np.array(box_list, dtype=np.float32)  # (N,4)
        widths = boxes[:, 2] - boxes[:, 0] + 1
        heights = boxes[:, 3] - boxes[:, 1] + 1
        areas = widths * heights
        return areas

    def area_range_reward(box_list, image_size, min_ratio=0.01, max_ratio=0.5):
        """
        Reward to check if each box's area is within a reasonable range.

        Args:
            box_list (list): list of boxes [x1, y1, x2, y2]
            image_size (tuple): (width, height) of the image
            min_ratio (float): minimum allowed area ratio w.r.t image (e.g., 0.01 means 1%)
            max_ratio (float): maximum allowed area ratio (e.g., 0.5 means 50%)

        Returns:
            float: 1.0 if all boxes pass the area check, else 0.0
        """
        if not box_list:
            return 0.0

        img_width, img_height = image_size
        img_area = img_width * img_height
        # 计算每个框的面积
        areas = compute_areas_numpy(box_list)
        # 计算每个框的面积占比
        area_ratios = areas / img_area
        # 检查每个框的面积占比是否在合理范围内
        if np.max(area_ratios) > max_ratio or np.min(area_ratios) < min_ratio:
            return 0.0
        return 1.0

    def json_format_reward(predict_str: str, strict: bool = False) -> float:
        def is_valid_format(predict_str: str) -> bool:
            try:
                # 先去掉 ```json 和 ```包围的东西
                if predict_str.startswith("```json"):
                    predict_str = predict_str[len("```json") :].strip()
                if predict_str.endswith("```"):
                    predict_str = predict_str[: -len("```")].strip()

                # 尝试解析成列表
                data = json.loads(predict_str)

                # 检查是不是一个列表
                if not isinstance(data, list):
                    return False

                # 列表里每一个元素应该是一个dict
                for item in data:
                    if not isinstance(item, dict):
                        return False

                    # 检查每个字典里有没有bbox_2d和label
                    if "bbox_2d" not in item or "label" not in item:
                        return False

                    bbox = item["bbox_2d"]
                    label = item["label"]

                    if not isinstance(bbox, list) or len(bbox) != 4:
                        return False
                    if not isinstance(label, str):
                        return False

                return True
            except Exception:
                return False

        return 1.0 if is_valid_format(predict_str) else 0.0

    def seg_box_l1_reward(
        predict_str: str, gt_bbox: List[int]
    ) -> Tuple[float, List[int]]:
        def l1_distance(box1, box2):
            return (
                abs(box1[0] - box2[0])
                + abs(box1[1] - box2[1])
                + abs(box1[2] - box2[2])
                + abs(box1[3] - box2[3])
            ) / 4

        try:
            # ground_truth = ground_truth.strip()
            # gt_box_pattern = r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>'
            # gt_match = re.search(gt_box_pattern, ground_truth)
            # if gt_match:
            #     gt_bbox = [int(gt_match.group(1)), int(gt_match.group(2)), int(gt_match.group(3)), int(gt_match.group(4))]

            json_pattern = r"{[^}]+}"
            json_match = re.search(json_pattern, predict_str)
            if json_match:
                data = json.loads(json_match.group(0))
                bbox_key = next(
                    (key for key in data.keys() if "bbox" in key.lower()), None
                )
                if bbox_key and len(data[bbox_key]) == 4:
                    content_bbox = data[bbox_key]
                    if l1_distance(content_bbox, gt_bbox) < 10:
                        return 1.0, content_bbox
        except Exception:
            pass
        return 0.0, None

    def refine_coord(coord: List[int], w_ratio: float, h_ratio: float) -> List[int]:
        """_summary_

        Args:
            coord (List[int]): [x, y]
            w_ratio (float): resized_width / original_width
            h_ratio (float): resized_height / original_height

        Returns:
            List[int]: corrected coordinates
        """
        return [int(coord[0] * w_ratio), int(coord[1] * h_ratio)]

    # question_type = kwargs['problem_type'][0]
    # if question_type != 'video segmentation':
    #     return 0
    # pred_results = [completion[0]["content"] for completion in completions]

    # contents = [completion[0]["content"] for completion in completions]
    # current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    gt_points_pattern = r"<points>\((\d+),(\d+)\),\((\d+),(\d+)\)</points>"
    gt_box_pattern = r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"

    # if isinstance(completions, str):
    #     completions = [completions]
    #     assert isinstance(gts, str), "pred_results and gts should be both str or both list"
    #     gts = [gts]

    # for pred, gt, (w_ratio, h_ratio) in zip(completions , gts, resize_ratio):

    pred = completion
    rh, rw = resized_hw
    w, h = kwargs["width"], kwargs["height"]
    w_ratio = rw / w
    h_ratio = rh / h
    object = kwargs["object"]
    target_annotation = kwargs["categories2ann"][object]  # coco format
    gt_bbox_list = [item["bbox"] for item in target_annotation]
    gt_bbox_list = convert_bbox_to_xyxy(gt_bbox_list)
    # apply resize ratio
    gt_bbox_list[:, 0] = gt_bbox_list[:, 0] * w_ratio
    gt_bbox_list[:, 1] = gt_bbox_list[:, 1] * h_ratio
    gt_bbox_list[:, 2] = gt_bbox_list[:, 2] * w_ratio
    gt_bbox_list[:, 3] = gt_bbox_list[:, 3] * h_ratio

    def convert_to_json(pred_str: str) -> dict:
        # Remove the ```json and ``` tags
        if pred_str.startswith("```json"):
            pred_str = pred_str[len("```json") :].strip()
        if pred_str.endswith("```"):
            pred_str = pred_str[: -len("```")].strip()

        # Parse the JSON string
        return json.loads(pred_str)

    def soft_load_json(json_str: str) -> list:
        json_pattern = r"{[^}]+}"
        json_match = re.findall(json_pattern, json_str)
        return_list = []
        for json_str in json_match:
            try:
                data = json.loads(json_str)
                return_list.append(data)
            except json.JSONDecodeError:
                print(f"Error decoding JSON: {json_str}")
        return return_list

    def soft_json_format_reward(pred_str: str) -> float:
        try:
            # 尝试解析成列表
            data = soft_load_json(pred_str)

            # 检查是不是一个列表
            if not isinstance(data, list):
                return 0.0

            # 列表里每一个元素应该是一个dict
            for item in data:
                if not isinstance(item, dict):
                    return 0.0

                # 检查每个字典里有没有bbox_2d和label
                if "bbox_2d" not in item or "label" not in item:
                    return 0.0

                bbox = item["bbox_2d"]
                label = item["label"]

                if not isinstance(bbox, list) or len(bbox) != 4:
                    return 0.0
                if not isinstance(label, str):
                    return 0.0

            return 1.0
        except Exception:
            print(f"Error parsing JSON: {pred_str}")
            return 0.0

    json_format_rewards = soft_json_format_reward(pred)
    try:
        # pred_json = convert_to_json(pred)
        pred_json = soft_load_json(pred)
    except Exception as e:
        print(f"Error converting prediction to JSON: {e}")
        print(f"Prediction string: {pred}")
        pred_json = None
    # pred_json is a  list of dict
    #     0 =
    # {'bbox_2d': [108, 79, 256, 242], 'label': 'earring-dense region'}
    # 1 =
    # {'bbox_2d': [462, 82, 636, 258], 'label': 'earring-dense region'}
    # 2 =
    # {'bbox_2d': [678, 127, 750, 258], 'label': 'earring-dense region'}
    if pred_json is not None and json_format_rewards > 0:
        try:
            pred_box_list = [item["bbox_2d"] for item in pred_json if "bbox_2d" in item]
            # 计算两者之间的IoU
            # 1. no overlap reward
            fast_no_overlap_rewards = fast_no_overlap_reward(
                pred_box_list, iou_threshold=0.5
            )

            # 2. area range reward
            box_area = compute_areas_numpy(
                pred_box_list
            )  # compute_iou(pred_box_list, pred_box_list)
            area_range_rewards = area_range_reward(
                pred_box_list, (rw, rh), min_ratio=0.001, max_ratio=0.8
            )

            # 3.use coverage rate to compute the reward
            gt_ins_num = len(gt_bbox_list)
            cov_num, coverage_rate = compute_coverage_rate(gt_bbox_list, pred_box_list)
            coverage_rate_rewards = coverage_rate

            # purity reward
            area_ratio = (box_area.sum()) / (rw * rh) + 1e-6
            purity_rewards = coverage_rate_rewards / area_ratio
            if coverage_rate_rewards < 0.3:
                purity_rewards = 0.0
        except Exception as e:
            print(f"Error cal reward: {e}")
            pred_box_list = []
            fast_no_overlap_rewards = 0.0
            area_range_rewards = 0.0
            coverage_rate_rewards = 0.0
            purity_rewards = 0.0

    else:
        pred_box_list = []
        fast_no_overlap_rewards = 0.0
        area_range_rewards = 0.0
        coverage_rate_rewards = 0.0
        purity_rewards = 0.0

    reward = (
        json_format_rewards
        + fast_no_overlap_rewards
        + area_range_rewards
        + coverage_rate_rewards
    )

    sol = f"resized_hw: {resized_hw}"
    output_path = None
    if os.getenv("DEBUG_MODE") != "true":
        return reward, sol, output_path
    kwargs["problem"] = "sod"
    # if an_points is not None:
    #     an_points = [(an_points[0]),(an_points[1])]
    # output_path = draw_example(resized_hw, kwargs['image'], kwargs['problem'], gt_bbox, gt_points, an_bbox, an_points, kwargs['problem_id'])
    # output_path = draw_example(resized_hw, kwargs['image'], kwargs['problem'], gt_bbox_list, pred_box_list, kwargs['problem_id'])

    return reward, sol, output_path


def soft_load_json(json_str: str) -> list:
    json_pattern = r"{[^}]+}"
    json_match = re.findall(json_pattern, json_str)
    return_list = []
    for json_str in json_match:
        try:
            data = json.loads(json_str)
            return_list.append(data)
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {json_str}")
    return return_list
