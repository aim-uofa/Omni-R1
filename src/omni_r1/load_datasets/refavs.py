import json
import os
import random
import re
import time
from pathlib import Path as pth
from typing import List

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from torch.utils.data import Dataset

REFAVS_TEMPLATE = (
    "Given a 10-second audio and a reference instruction: [ref_prompt], which involves temporal and audio-related behavior, "
    "first analyze the objects in the image that are producing sound — including both human voices and instrument sounds. "
    "Then, based on the audio content, identify the exact object [ref_prompt] in the image that matches the audio. "
    "Next, simplify the identified object into a short and clear visual grounding description, "
    "so that it can be unambiguously recognized in a single image without relying on audio. "
    "Avoid using temporal expressions (such as 'playing', or 'moving'), "
    "and instead describe direct visible visual cues such as clothing, pose, position, or grouping. "
    "Write your reasoning in <think>...</think> and the final result in <answer>...</answer>."
)


class REFAVS(Dataset):
    def __init__(self, dataset_path: str, pick: int = None, split="train"):
        # metadata: train/test/val
        self.data_dir = pth(dataset_path).resolve()  # dataset_path
        if not self.data_dir.exists():
            raise ValueError(f"Dataset folder {str(self.data_dir)} does not exist.")

        meta_path = self.data_dir / "metadata.csv"
        metadata = pd.read_csv(meta_path, header=0)
        self.split = split
        self.metadata = metadata[metadata["split"] == split]  # split= train,test,val.

        self.media_path = self.data_dir / "media"
        self.label_path = self.data_dir / "gt_mask"
        self.frame_num = 10

        if pick is not None:
            self.selected_metadata = self.mask_samples(pick)
        else:
            self.selected_metadata = self.metadata

    def mask_samples(self, num_samples):
        """
        Randomly sample a subset of metadata for training.
        """
        if len(self.metadata) > num_samples:
            selected_indices = random.sample(range(len(self.metadata)), num_samples)
        else:
            print(
                f"Metadata contains only {len(self.metadata)} samples, no sampling applied."
            )
        return self.metadata.iloc[selected_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.selected_metadata)

    def __getitem__(self, idx):
        idx = idx % len(self.selected_metadata)

        df_one_video = self.selected_metadata.iloc[idx]
        vid, uid, fid, exp = (
            df_one_video["vid"],
            df_one_video["uid"],
            df_one_video["fid"],
            df_one_video["exp"],
        )  # uid for vid.
        vid = uid.rsplit("_", 2)[0]  # TODO: use encoded id.

        p = self.label_path / f"{vid}/fid_{fid}"
        if not p.exists():
            return self.__getitem__(
                random.randint(0, len(self.selected_metadata) - 1)
            )  # random sample if not exist

        img_recs = []
        mask_recs = []

        rec_audio = self.media_path / f"{vid}/audio.wav"
        rec_text = exp
        frames = []

        # feat_aud = self.get_audio_emb(rec_audio)
        # feat_text = self.get_text_emb(rec_text)

        for _idx in range(self.frame_num):  # set frame_num as the batch_size
            # frame
            path_frame = f"{self.media_path}/{vid}/frames/{_idx}.jpg"  # image
            image = Image.open(path_frame)
            frames.append(path_frame)
            image_wh = image.size  # W, H

            # mask label
            path_mask = self.label_path / f"{vid}/fid_{fid}/0000{_idx}.png"  # new
            mask_cv2 = cv2.imread(str(path_mask))
            mask_cv2 = cv2.resize(mask_cv2, image_wh)
            mask_cv2 = cv2.cvtColor(mask_cv2, cv2.COLOR_BGR2GRAY)
            gt_binary_mask = np.array(mask_cv2 > 0)

            # video frames collect
            img_recs.append(image)
            mask_recs.append(gt_binary_mask)

        masks = np.stack(mask_recs, axis=0)  # [frames, H, W]
        points = generate_points(masks)
        bboxs = generate_bbox(masks)
        prompt = self._make_prompt(vid, img_recs, rec_audio, rec_text)

        return {
            "data_type": "video",
            "problem_type": "refavs",
            "problem_id": vid,
            "problem": rec_text,
            "frames": frames,
            "masks": masks[None, :, :, :],
            "points": points[None, :, :],
            "bboxs": bboxs[None, :, :],
            "prompt": prompt,
        }

    def _make_prompt(self, vid, img_list: List[Image.Image], audio_path: str, exp: str):
        """
        Make a prompt for the model.
        """
        content = []
        # interpolation_idx = np.linspace(0, len(img_list)-1, 2 * len(img_list)).astype(int).tolist()
        # interpolated_imgs = [img_list[i] for i in interpolation_idx]
        # video_path = self._gen_video(vid, interpolated_imgs, audio_path, VIDEO_CACHE_DIR, 2)
        #
        content.append({"type": "audio", "audio": "file://" + str(audio_path)})
        # content.append({
        #     "type": "video",
        #     "video": video_path, # interpolated_imgs
        #     "max_pixels": 81 * 28 * 28
        # })
        content.append({"type": "image", "image": img_list[0]})
        content.append(
            {
                "type": "text",
                "text": REFAVS_TEMPLATE.replace("[ref_prompt]", f"[{exp}]"),
            }
        )

        prompt = [{"role": "user", "content": content}]
        return prompt


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


def refavs_grounding_prompt(ans: str, **kwargs):
    text = "Please find the bounding box of '{Question}' with json format".format(
        Question=ans.lower().strip()
    )
    frames = kwargs.get("frames", None)
    assert frames is not None, "Image should not be None"
    input = [
        {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": frames[0],
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]
        }
    ]
    return input


def refavs_grounding_process_reward(
    idx: int,
    raw_problem: str,
    input: str,
    completion: str,
    grounding_arg: dict,
    raw_args: dict,
):
    bboxs = raw_args.get("bboxs", None)
    assert bboxs is not None, "bboxs should not be None"

    img_resized_hw = grounding_arg.get("img_resized_hw", None)[0]
    img_ori_hw = grounding_arg.get("img_ori_hw", None)[0]
    if img_resized_hw is None or img_ori_hw is None:
        raise ValueError("img_resized_hw or img_ori_hw is None")

    h_ratio = img_resized_hw[0] / float(img_ori_hw[0])
    w_ratio = img_resized_hw[1] / float(img_ori_hw[1])

    # bboxs = np.stack(bboxs, axis=0)
    # points = np.stack(points, axis=0)
    scale = np.array([w_ratio, h_ratio, w_ratio, h_ratio]).astype(np.float32)
    scaled_bboxs = bboxs[:, :1].astype(np.float32) * scale

    json_pattern = r"({[\s\S]*?})"
    js_match = re.search(json_pattern, completion)
    pred_bbox = []

    try:
        js_result = json.loads(js_match.group(0))
        assert (
            isinstance(js_result["bbox_2d"], list) and len(js_result["bbox_2d"]) == 4
        ), "bbox_2d should have 4 elements"
        pred_bbox.append(js_result["bbox_2d"])
    except Exception:
        pred_bbox.append([-1, -1, -1, -1])
    pred_bbox = np.array(pred_bbox, dtype=np.float32)[None, :, :]
    iou_value = np.sum(iou(pred_bbox, scaled_bboxs))

    if os.getenv("PLOG") == "true":
        write_log(
            idx,
            raw_problem,
            inputs=[input],
            completions=[completion],
            arg_dict=[grounding_arg],
            iou_value=iou_value,
            pred_bboxs=pred_bbox,
            scaled_bboxs=scaled_bboxs,
        )

    return iou_value


def iou(pred_bboxs: np.array, gt_bboxs: np.array) -> np.array:
    """Calculate Intersection over Union (IoU) between predicted and ground truth bounding boxes.
    This function computes the IoU between two sets of bounding boxes represented as arrays.
    IoU is calculated as the area of intersection divided by the area of union.
    Args:
        pred_bboxs (np.array): Predicted bounding boxes with shape (batch_size, num_boxes, 4).
                              Each box is represented as [x1, y1, x2, y2] where (x1, y1) is
                              the top-left corner and (x2, y2) is the bottom-right corner.
        gt_bboxs (np.array): Ground truth bounding boxes with the same shape as pred_bboxs.

    Returns:
        np.array: IoU matrix with shape (batch_size, num_boxes) where each element is the
                  IoU score between the corresponding predicted and ground truth boxes.
    """

    assert pred_bboxs.shape == gt_bboxs.shape, (
        "pred_bboxs and gt_bboxs should have the same shape"
    )

    inter_x1 = np.maximum(pred_bboxs[:, :, 0], gt_bboxs[:, :, 0])
    inter_y1 = np.maximum(pred_bboxs[:, :, 1], gt_bboxs[:, :, 1])
    inter_x2 = np.minimum(pred_bboxs[:, :, 2], gt_bboxs[:, :, 2])
    inter_y2 = np.minimum(pred_bboxs[:, :, 3], gt_bboxs[:, :, 3])

    inter_w = np.maximum(0, inter_x2 - inter_x1 + 1)
    inter_h = np.maximum(0, inter_y2 - inter_y1 + 1)

    inter_area = inter_w * inter_h

    pred_area = (pred_bboxs[:, :, 2] - pred_bboxs[:, :, 0] + 1) * (
        pred_bboxs[:, :, 3] - pred_bboxs[:, :, 1] + 1
    )
    gt_area = (gt_bboxs[:, :, 2] - gt_bboxs[:, :, 0] + 1) * (
        gt_bboxs[:, :, 3] - gt_bboxs[:, :, 1] + 1
    )

    union_area = pred_area + gt_area - inter_area

    iou_matrix = inter_area / union_area

    return iou_matrix


def write_log(
    nidx: int,
    raw_problem: str,
    inputs: list,
    completions: list,
    arg_dict: list,
    iou_value: str,
    pred_bboxs: np.array,
    scaled_bboxs: np.array,
):
    """将输入和输出写入日志文件"""

    tstr = time.strftime("%Y%m%d-%H%M%S")

    for i, (input, completion, data_info) in enumerate(
        zip(inputs, completions, arg_dict)
    ):
        img_resized_hw = data_info.get("img_resized_hw", None)[0]
        img_ori_hw = data_info.get("img_ori_hw", None)[0]
        if img_resized_hw is None or img_ori_hw is None:
            raise ValueError("img_resized_hw or img_ori_hw is None")

        # assume obj_num is 1
        pred_bbox = pred_bboxs[:, i]
        scaled_bbox = scaled_bboxs[:, i]
        img = Image.open(input["prompt"][0]["content"][0]["image"]).resize(
            (img_resized_hw[1], img_resized_hw[0]), Image.Resampling.BICUBIC
        )

        # Draw bounding boxes on the image
        draw = ImageDraw.Draw(img)
        # Define colors for different purposes
        bbox_color = (255, 0, 0)  # Red for predictions
        gt_color = (0, 255, 0)  # Green for ground truth

        # Draw predicted bbox
        if not (pred_bbox == -1).all():
            x1, y1, x2, y2 = tuple(pred_bbox.squeeze().tolist())
            draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=2)
            # draw.text((0, 0), "Pred", fill=text_color)

        # Draw ground truth bbox
        if not (scaled_bbox == -1).all():
            x1, y1, x2, y2 = tuple(scaled_bbox.squeeze().tolist())
            draw.rectangle([x1, y1, x2, y2], outline=gt_color, width=2)
            # draw.text((0, 0), "GT", fill=text_color)

        # Save the annotated image for debugging
        output_dir = pth(os.getenv("LOG_PATH")).resolve() / (
            tstr + "ROLL_" + str(nidx) + "_R" + str(torch.distributed.get_rank())
        )
        os.makedirs(output_dir, exist_ok=True)
        img.save(os.path.join(output_dir, f"iou_{iou_value:.2f}.jpg"))

        with open((output_dir / "info.txt"), "a", encoding="utf-8") as f:
            f.write(f"--------------------Frame {i}:--------------------\n")
            f.write(f"Raw Problem:\n {raw_problem}\n")
            f.write(f"File Path: {input['prompt'][0]['content'][0]['image']}\n")
            f.write(f"Input:\n {input}\n")
            f.write(f"Completion:\n {completion}\n")
