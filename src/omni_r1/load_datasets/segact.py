import json
import math
import os
import re
from pathlib import Path as pth
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from datasets import Dataset, load_dataset, load_from_disk
from pycocotools import mask as maskUtils
from scipy.ndimage import distance_transform_edt

VISUALIZATION_PATH = pth("./visualizations")

QUESTION_TEMPLATE = (
    "Please find '{Question}' with bbox and points."
    "Compare the difference between objects and find the most closely matched one."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format."
    "i.e., <think> thinking process here </think>"
    "<answer>{Answer}</answer>"
)

answer = (
    '{ "bbox_2d" : [10,100,200,210], "points_1" : [30,110], "points_2" : [35,180] }'
)


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


MAX_DATASET_SIZE = 9000
import random


class SegActDataset(Dataset):
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
        self.raw_dataset = self.raw_dataset["data"]
        random.shuffle(self.raw_dataset)
        self.raw_dataset = self.raw_dataset[:MAX_DATASET_SIZE]
        # get data dir
        self.data_dir = os.path.dirname(self.dataset_path)
        self.dataset_name = self.dataset_path.split("/")[-1].split(".")[0].split("_")[0]
        self.data_dir = self.data_dir + "/images" + "/"

        print(f"Loaded {len(self.raw_dataset)} samples from {self.dataset_path}")
        # self.rank0_load(batched= batch_size > 1, batch_size=batch_size, num_proc=num_proc)
        # load additional gt_mask to compute reward
        # additional_json = '/home/zmz/code/SimpleClick/data/record_trace/refcoco+_train.json'
        # with open(additional_json, 'r') as f:
        #     self.additional_gt = json.load(f)
        # self.additional_gt['data'][0]
        # {'image_name': 519404, 'height': 480, 'width': 640, 'gt_ann': {'object_id': 0, 'caption': [...], 'segmentation': {...}, 'bbox': [...], 'area': 57677},
        # self.additional_gt['data'][0]['gt_ann']
        # {'object_id': 0, 'caption': ['two woman one in black eatting and the other has a white shirt at the desk', 'woman in white shirt looking down at laptop computer'], 'segmentation': {'counts': 'j3i3U;\\3cL4M3M3L4M3L4M3L4M3L4M3M3L4M3M3M3M3N2M3N2M3M3N2M3N2M3M3N2M3N2M3M3N2M3N2O1O1N2...0O1O100O1N2O1N2O1N2O1N2O0O2M2N3L6K7H7JZfk5', 'size': [...]}, 'bbox': [0.0, 45.0, 239.0, 410.0], 'area': 57677}
        # self.imgid2gt = {}
        # self.caption2gt = {}
        # for item in self.additional_gt['data']:
        #     imgid = item['image_name']
        #     gt_bbox = item['gt_ann']['bbox']
        #     width = item['width']
        #     height = item['height']
        #     gt_segmentation = item['gt_ann']['segmentation']
        #     gt_caption_list = item['gt_ann']['caption']
        #     for gt_caption in gt_caption_list:
        #         if imgid not in self.imgid2gt:
        #             self.imgid2gt[imgid] = {}
        #         if gt_caption not in self.imgid2gt[imgid]:
        #             self.imgid2gt[imgid][gt_caption] = {}
        #         self.imgid2gt[imgid][gt_caption]['bbox'] = gt_bbox
        #         self.imgid2gt[imgid][gt_caption]['segmentation'] = gt_segmentation
        #         self.imgid2gt[imgid][gt_caption]['area'] = item['gt_ann']['area']
        #         self.imgid2gt[imgid][gt_caption]['object_id'] = item['gt_ann']['object_id']
        #         self.imgid2gt[imgid][gt_caption]['width'] = item['width']
        #         self.imgid2gt[imgid][gt_caption]['height'] = item['height']
        #         if gt_caption not in self.caption2gt:
        #             self.caption2gt[gt_caption] = []
        #             temp = {}
        #             temp['image_name'] = imgid
        #             temp['bbox'] = gt_bbox
        #             temp['segmentation'] = gt_segmentation
        #             temp['area'] = item['gt_ann']['area']
        #             temp['object_id'] = item['gt_ann']['object_id']
        #             temp['width'] = item['width']
        #             temp['height'] = item['height']
        #             self.caption2gt[gt_caption].append(temp)
        #         else:
        #             print(f"Warning: {gt_caption} already exists in caption2gt")
        #             temp = {}
        #             temp['image_name'] = imgid
        #             temp['bbox'] = gt_bbox
        #             temp['segmentation'] = gt_segmentation
        #             temp['area'] = item['gt_ann']['area']
        #             temp['object_id'] = item['gt_ann']['object_id']
        #             temp['width'] = item['width']
        #             temp['height'] = item['height']
        #             self.caption2gt[gt_caption].append(temp)

        # self.dataset = load_dataset(self.dataset_path)
        # self.load_data(batched= batch_size > 1, batch_size=batch_size, num_proc=num_proc)

    def get_image_path(self, image_id, dataset_name):
        if dataset_name == "refclef":
            return f"saiapr_tc-12/{str(image_id // 1000).zfill(2)}/images/{str(image_id)}.jpg"
        elif (
            dataset_name == "refcoco"
            or dataset_name == "refcoco+"
            or dataset_name == "refcocog"
        ):
            return f"mscoco/images/train2014/COCO_train2014_000000{str(image_id).zfill(6)}.jpg"
        else:
            raise ValueError(f"Unknown dataset name {dataset_name}")

    def annToMask(self, mask_ann, h, w):
        if mask_ann is None:
            return np.zeros((h, w), dtype=np.uint8)

        if isinstance(mask_ann, list):
            rles = maskUtils.frPyObjects(mask_ann, h, w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann["counts"], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, h, w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

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
        idx = idx if isinstance(idx, list) else [idx]
        items_list = []
        for i in idx:
            items = self.raw_dataset[i]
            items_list.append(items)
        items = {k: [d[k] for d in items_list] for k in items_list[0].keys()}
        # items = self.raw_dataset[idx]
        bs = len(idx)

        items["prompt"] = []
        items["data_type"] = ["image"] * bs
        items["problem_type"] = ["image-segmentation"] * bs
        items["problem_id"] = items["image_name"]
        # ['refcocog_16521']
        # items['image_id'] = [ problem_id.split('_')[1] for problem_id in items['problem_id']]
        items["bboxs"] = []
        items["masks"] = []
        items["points"] = []
        # 'solution': '<box>(0,457),(374,672)</box><points>(50,592),(144,601)</points>'
        items["solution"] = []
        items["image"] = []
        for i in range(bs):
            image_id = items["image_name"][i]
            file_name = self.get_image_path(image_id, self.dataset_name)
            image_path = os.path.join(self.data_dir, file_name)
            # file_name = items['gt_ann'][i]
            gt_ann = items["gt_ann"][i]
            gt_mask = items["gt_ann"][i]["segmentation"]
            gt_mask_array = self.annToMask(
                gt_mask, items["height"][i], items["width"][i]
            )
            gt_box_xyhw = items["gt_ann"][i]["bbox"]
            gt_box_xxyy = [
                gt_box_xyhw[0],
                gt_box_xyhw[1],
                gt_box_xyhw[0] + gt_box_xyhw[2],
                gt_box_xyhw[1] + gt_box_xyhw[3],
            ]
            gt_box_xxyy = [
                gt_box_xxyy[0],
                gt_box_xxyy[1],
                gt_box_xxyy[2],
                gt_box_xxyy[3],
            ]
            gt_box_xxyy = np.array(gt_box_xxyy)
            gt_points = generate_points(gt_mask_array[None])

            # items['image'][i] = pth(self.dataset_path).parent / file_name
            caption_list = items["gt_ann"][i]["caption"]
            # 'solution': '<box>(0,457),(374,672)</box><points>(50,592),(144,601)</points>'
            solution = f"<box>({gt_box_xxyy[0]},{gt_box_xxyy[1]}),({gt_box_xxyy[2]},{gt_box_xxyy[3]})</box><points>({gt_box_xyhw[0]},{gt_box_xyhw[1]}),({gt_box_xyhw[0] + gt_box_xyhw[2]},{gt_box_xyhw[1] + gt_box_xyhw[3]})</points>"
            items["solution"].append(solution)
            question = random.choice(caption_list)
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                            "min_pixels": 1024 * 28 * 28,
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

            items["prompt"].append(prompt)
            items["image"].append(image_path)
            # items['problem_id'].append(f"{self.dataset_name}_{image_id}")
            items["masks"].append(gt_mask_array[None])  # 1x1xh*w
            items["bboxs"].append(gt_box_xxyy[None])
            items["points"].append(gt_points)
        f_items = {k: v[0] for k, v in items.items()}
        return f_items


def draw_example(
    resized_hw, image, problem: str, gt_bbox, gt_points, an_bbox, an_points, id: str
):
    w, h = image.size

    rh, rw = resized_hw

    resized_image = image.resize((rw, rh))
    # Convert PIL image to CV2 format (RGB to BGR)
    resized_image_cv = cv2.cvtColor(np.array(resized_image), cv2.COLOR_RGB2BGR)
    ans_img = resized_image_cv.copy()
    # Draw the bounding box
    x1, y1, x2, y2 = gt_bbox
    cv2.rectangle(
        resized_image_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
    )
    # Draw the points
    for point in gt_points:
        x, y = point
        cv2.circle(resized_image_cv, (int(x), int(y)), 5, (0, 0, 255), -1)

    # Add problem text as annotation
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(resized_image_cv, problem, (10, 30), font, 0.7, (255, 0, 0), 2)

    if an_bbox is not None:
        x1, y1, x2, y2 = an_bbox
        cv2.rectangle(ans_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    if an_points is not None:
        for point in an_points:
            x, y = point
            cv2.circle(ans_img, (int(x), int(y)), 5, (0, 255, 0), -1)

    # Save the visualization
    output_dir = pth(VISUALIZATION_PATH).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"{id}"
    output_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path / "gt.jpg"), resized_image_cv)
    cv2.imwrite(str(output_path / "ans.jpg"), ans_img)
    return str(output_path)


def segzero_reward(
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

    def seg_iou_reward(predict_str: str, gt_bbox: List[int]) -> float:
        def iou(box1, box2):
            inter_x1 = max(box1[0], box2[0])
            inter_y1 = max(box1[1], box2[1])
            inter_x2 = min(box1[2], box2[2])
            inter_y2 = min(box1[3], box2[3])
            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
            else:
                inter = 0
            area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
            area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
            union = area1 + area2 - inter
            return float(inter) / union

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
                    return iou(content_bbox, gt_bbox)
        except Exception:
            pass
        return 0.0

    def seg_segmentation_format_reward(predict_str: str, strict: bool = False) -> float:
        def is_valid_format(predict_str: str) -> bool:
            try:
                json_match = re.search(r"{[^}]+}", predict_str)
                if not json_match:
                    return False
                json_str = json_match.group(0)
                data = json.loads(json_str)

                if not strict:
                    bbox_key = None
                    points_keys = []

                    for key in data.keys():
                        if "bbox" in key.lower() and bbox_key is None:
                            bbox_key = key
                        elif "point" in key.lower():
                            points_keys.append(key)

                    if not (bbox_key and len(points_keys) >= 2):
                        return False

                    bbox = data[bbox_key]
                    if len(bbox) != 4:
                        return False

                    for key in points_keys[:2]:
                        if len(data[key]) != 2:
                            return False
                else:
                    # check the required keys
                    required_keys = ["bbox_2d", "points_1", "points_2"]
                    for key in required_keys:
                        if key not in data:
                            return False

                    # check the format of the value
                    bbox = data["bbox_2d"]
                    if not isinstance(bbox, list) or len(bbox) != 4:
                        return False

                    points_1 = data["points_1"]
                    points_2 = data["points_2"]
                    if not isinstance(points_1, list) or len(points_1) != 2:
                        return False
                    if not isinstance(points_2, list) or len(points_2) != 2:
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
                    return l1_distance(content_bbox, gt_bbox), content_bbox
        except Exception:
            pass
        return 0.0, None

    def seg_point_l1_reward(
        predict_str: str, gt_points: List[List[int]]
    ) -> Tuple[float, Optional[List[int]]]:
        def points_in_box(point, bbox):
            return bbox[0] <= point[0] <= bbox[2] and bbox[1] <= point[1] <= bbox[3]

        def points_distance(points1, points2):
            dist1 = math.sqrt(
                (points1[0][0] - points2[0][0]) ** 2
                + (points1[0][1] - points2[0][1]) ** 2
            ) + math.sqrt(
                (points1[1][0] - points2[1][0]) ** 2
                + (points1[1][1] - points2[1][1]) ** 2
            )

            dist2 = math.sqrt(
                (points1[0][0] - points2[1][0]) ** 2
                + (points1[0][1] - points2[1][1]) ** 2
            ) + math.sqrt(
                (points1[1][0] - points2[0][0]) ** 2
                + (points1[1][1] - points2[0][1]) ** 2
            )
            return min(dist1, dist2) / 2

        try:
            # gt_points_pattern = r'<points>\((\d+),(\d+)\),\((\d+),(\d+)\)</points>'
            # gt_match = re.search(gt_points_pattern, ground_truth)
            # if gt_match:
            #     gt_points = [[int(gt_match.group(1)), int(gt_match.group(2))], [int(gt_match.group(3)), int(gt_match.group(4))]]
            json_pattern = r"{[^}]+}"
            json_match = re.search(json_pattern, predict_str)
            if json_match:
                data = json.loads(json_match.group(0))
                bbox_key = next(
                    (key for key in data.keys() if "bbox" in key.lower()), None
                )
                if bbox_key and len(data[bbox_key]) == 4:
                    content_bbox = data[bbox_key]
                points_keys = [key for key in data.keys() if "points" in key.lower()][
                    :2
                ]
                if len(points_keys) == 2:
                    point1 = data[points_keys[0]]
                    point2 = data[points_keys[1]]
                    point1 = [int(point1[0]), int(point1[1])]
                    point2 = [int(point2[0]), int(point2[1])]
                    if points_in_box(point1, content_bbox) and points_in_box(
                        point2, content_bbox
                    ):
                        return points_distance([point1, point2], gt_points), (
                            point1,
                            point2,
                        )
        except Exception:
            pass  # Continue to next verification method if this fails
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
    w, h = kwargs["image"].size
    w_ratio = rw / w
    h_ratio = rh / h

    seg_format_reward = seg_segmentation_format_reward(pred, strict)

    gt_bbox_match = re.search(gt_box_pattern, gt)
    if gt_bbox_match:
        gt_bbox = [
            int(gt_bbox_match.group(1)),
            int(gt_bbox_match.group(2)),
            int(gt_bbox_match.group(3)),
            int(gt_bbox_match.group(4)),
        ]

    # Refine bbox coordinates but keep as flat list [x1, y1, x2, y2]
    x1, y1 = refine_coord(gt_bbox[:2], w_ratio, h_ratio)
    x2, y2 = refine_coord(gt_bbox[2:], w_ratio, h_ratio)
    gt_bbox = [x1, y1, x2, y2]
    iou = seg_iou_reward(pred, gt_bbox)
    box_l1, an_bbox = seg_box_l1_reward(pred, gt_bbox)

    gt_points_match = re.search(gt_points_pattern, gt)
    if gt_points_match:
        gt_points = [
            [int(gt_points_match.group(1)), int(gt_points_match.group(2))],
            [int(gt_points_match.group(3)), int(gt_points_match.group(4))],
        ]

    gt_points = [
        refine_coord(gt_points[0], w_ratio, h_ratio),
        refine_coord(gt_points[1], w_ratio, h_ratio),
    ]
    point_l1, an_points = seg_point_l1_reward(pred, gt_points)

    # SegZero Reward seems not improving the performance
    iou_reward = max(0.0, iou - 0.5) * 2
    point_l1_reward = min(1.0, max(0.0, (150.0 - point_l1) * 0.02))
    box_l1_reward = min(1.0, max(0.0, (30.0 - box_l1) * 0.05))

    reward = iou_reward + point_l1_reward + box_l1_reward + seg_format_reward

    sol = f"resized_hw: {resized_hw}, 'bbox': {gt_bbox}, 'points_1': {gt_points[0]}, 'points_2': {gt_points[1]}"

    if os.getenv("LOG_MODE") != "true":
        return reward, None, None

    if an_points is not None:
        an_points = [(an_points[0]), (an_points[1])]
    output_path = draw_example(
        resized_hw,
        kwargs["image"],
        kwargs["problem"],
        gt_bbox,
        gt_points,
        an_bbox,
        an_points,
        kwargs["problem_id"],
    )

    return reward, sol, output_path
