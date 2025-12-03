import json
import math
import os
import re
from pathlib import Path as pth
from typing import List, Optional, Tuple

import cv2
import numpy as np
from datasets import Dataset, load_dataset

VISUALIZATION_PATH = pth("./visualizations")

QUESTION_TEMPLATE = (
    "Please find '{Question}' with bbox and points."
    "Compare the difference between objects and find the most closely matched one."
    "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    "Output the one bbox and points of two largest inscribed circles inside the interested object in JSON format."
    "i.e., <think> thinking process here </think>"
    "<answer>{Answer}</answer>"
)


answer = '{"bbox_2d": [x1, y1, x2, y2], "points_1": [x3, y3], "points_2": [x4, y4]}'


class SegZeroDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        *args,
        **kwargs,
    ):
        dataset_path = pth(dataset_path).resolve()
        if not dataset_path.exists():
            raise ValueError(f"Seg-Zero Dataset path {dataset_path} does not exist.")
        self.dataset_path = str(dataset_path)
        self.raw_dataset = load_dataset(self.dataset_path)["train"]

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
        idx = idx % len(self.raw_dataset)
        item = self.raw_dataset[idx]

        item["data_type"] = "image"
        item["problem_type"] = "image-segmentation"
        item["problem_id"] = item["id"]

        question = item["problem"]
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": item["image"],
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
        item["prompt"] = prompt

        return item


def draw_example(
    resized_hw, image, problem: str, gt_bbox, gt_points, an_bbox, an_points, id: str
):
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

    output_path = output_dir / f"{id}"
    output_path.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path / "gt.jpg"), resized_image_cv)
    cv2.imwrite(str(output_path / "ans.jpg"), ans_img)
    return output_path


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
        return 999, None

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
        return 999, None

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

    gt_points_pattern = r"<points>\((\d+),(\d+)\),\((\d+),(\d+)\)</points>"
    gt_box_pattern = r"<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>"

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
    iou_reward = iou_reward + 0.4 if iou > 0.5 else iou_reward

    point_l1_reward = min(1.0, max(0.0, (150.0 - point_l1) * 0.02))
    box_l1_reward = min(1.0, max(0.0, (30.0 - box_l1) * 0.05))

    reward = iou_reward + point_l1_reward + box_l1_reward + seg_format_reward

    sol = f"resized_hw: {resized_hw}, 'bbox': {gt_bbox}, 'points_1': {gt_points[0]}, 'points_2': {gt_points[1]}"

    if os.getenv("LOG_MODE") != "true":
        return reward, None, None

    if an_points is not None:
        an_points = [(an_points[0]), (an_points[1])]

    output_path = None
    if os.getenv("PLOG") == "true":
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
