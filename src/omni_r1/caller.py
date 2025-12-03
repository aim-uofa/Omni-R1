import os
import shutil
import tempfile

import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

SAM_PATH = os.getenv("SAM_PATH", "sam2.1-hiera-large/sam2.1_hiera_large.pt")


class Caller:
    __instance = None

    def __new__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(Caller, cls).__new__(cls)
        return cls.__instance

    def __init__(self, server_instance=None):
        if not hasattr(self, "initialized"):
            assert server_instance is not None, (
                "Server instance cannot be None on the Creation of Caller Instance."
            )
            self.server_instance = server_instance
            self.initialized = True

            local_rank = torch.distributed.get_rank()
            SAM_TYPE = torch.bfloat16
            device = (
                f"cuda:{local_rank % torch.cuda.device_count()}"
                if torch.cuda.is_available()
                else "cpu"
            )
            device = torch.device(device)

            if os.getenv("USE_LOCAL_SAM") == "1":
                self.predictor = build_sam2_video_predictor(
                    config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                    ckpt_path=SAM_PATH,
                    torch_dtype=SAM_TYPE,
                    device=device,
                )
                print("SAM2 loaded...")
            else:
                self.predictor = None
                print("SAM2 not loaded, using server instance for inference...")

    def __call__(self, requests: list):
        """_summary_

        Args:
            request (list): list of dicts with keys: video_dir, bbox, frame_names

        Returns:
            _type_: _description_
        """

        packed_requests, packed_other_args, obj_id_mapping, reverse_mapping = (
            self.pack_queue(requests)
        )
        if self.predictor is None:
            responses = self.server_instance(
                **packed_requests,
                host=os.getenv("SAM_HOST", "127.0.0.1"),
                port=int(os.getenv("SAM_PORT", "32224")),
            )  # host_1
            # if responses is empty dict, return None
            if len(responses) == 0:
                return None
        else:
            torch.cuda.empty_cache()
            responses = seg_bidirectional_from_bbox(
                self.predictor,
                video_dir=packed_requests["video_dir"],
                frame_names=packed_requests["frame_names"],
                permuted_pred_bboxs_list=packed_requests["bbox"],
            )
            torch.cuda.empty_cache()

        unpacked_responses = self.unpack_queue(
            responses, packed_other_args, reverse_mapping
        )
        return unpacked_responses

    def pack_queue(self, requests):
        other_args = []
        request_dicts = []
        for pack in requests:
            args = {}
            request_dict = None
            for k, v in pack.items():
                if k == "request":
                    request_dict = v
                else:
                    args[k] = v

            assert request_dict is not None, "Request dict cannot be None"
            request_dicts.append(request_dict)
            other_args.append(args)

        packed_requests = {}
        bboxs_list = []
        for k, v in request_dicts[0].items():
            if k != "bbox":
                packed_requests[k] = v
            else:
                for req in request_dicts:
                    bboxs_list.append(req["bbox"])

        bboxs, obj_id_mapping, reverse_mapping = aggregate_bboxes_and_treat_each_as_obj(
            bboxs_list
        )
        packed_requests["bbox"] = bboxs

        return packed_requests, other_args, obj_id_mapping, reverse_mapping

    def unpack_queue(self, responses, packed_other_args, reverse_mapping):
        """_summary_

        Args:
            responses (dict): the segmentation results {keys: obj_id, {frame_id: {mask}}}
            packed_other_args (_type_): _description_
            obj_id_mapping (_type_): _description_
        """
        groups = {}
        unpacked_responses = []

        each_obj_as_obj = False
        for obj_id, masks in responses.items():
            group_idx, original_obj_id, entry_idx, frame_id = (
                *reverse_mapping[obj_id],
                None,
                None,
            )[:4]
            if entry_idx is None:
                if group_idx not in groups.keys():
                    groups[group_idx] = {}
                groups[group_idx][original_obj_id] = masks

            else:
                each_obj_as_obj = True
                if group_idx not in groups.keys():
                    groups[group_idx] = []
                groups[group_idx].append(masks)

        for i in range(len(packed_other_args)):
            response = {"response": groups[i]}
            for k, v in packed_other_args[i].items():
                response[k] = v
            unpacked_responses.append(response)
        return unpacked_responses


def aggregate_bboxes_and_treat_each_as_obj(pred_bboxs_list):
    """
    聚合 bbox，将每个 bbox 都视为独立对象，并分配唯一的全局 ID。

    Args:
        pred_bboxs_list: 包含多个 permuted_pred_bboxs_list 的列表，每个 permuted_pred_bboxs_list 格式为 [[obj_id, frame_id, bbox], ...]

    Returns:
        aggregated_bboxes: 聚合后的 bbox 列表，格式为 [[new_obj_id, frame_id, bbox], ...]
        obj_id_mapping: 字典，保存原始 (group_id, obj_id, frame_id, bbox_index) 到新 obj_id 的映射
        rever_mapping: 字典，保存新 obj_id 到原始 (group_id, obj_id) 的映射
    """
    aggregated_bboxes = []  # 聚合后的 bbox 列表
    obj_id_mapping = {}  # 保存原始信息到新 obj_id 的映射
    rever_mapping = {}
    new_obj_id_counter = 0  # 新的连续 obj_id 的计数器

    # 遍历 pred_bboxs_list 中的每个组（permuted_pred_bboxs_list）
    for group_idx, permuted_pred_bboxs_list in enumerate(pred_bboxs_list):
        # 遍历每组中的每个 entry
        for entry_idx, entry in enumerate(permuted_pred_bboxs_list):
            original_obj_id, frame_id, bbox = entry[:]

            # 为每个 bbox 创建唯一的键
            # 使用(group_idx, entry_idx, original_obj_id, frame_id)确保每个 bbox 都是唯一的
            unique_key = (group_idx, entry_idx, original_obj_id, frame_id)

            # 分配一个新的 obj_id
            obj_id_mapping[unique_key] = new_obj_id_counter

            # 保存反向映射，只保留 group_idx 和 original_obj_id 用于后期恢复
            rever_mapping[new_obj_id_counter] = (
                group_idx,
                original_obj_id,
                entry_idx,
                frame_id,
            )

            # 增加计数器
            new_obj_id_counter += 1

            # 将新 obj_id 和对应的 bbox 添加到聚合的结果列表中
            aggregated_bboxes.append([new_obj_id_counter - 1, frame_id, bbox])

    return aggregated_bboxes, obj_id_mapping, rever_mapping


def seg_bidirectional_from_bbox(
    predictor,
    video_dir,
    frame_names,
    permuted_pred_bboxs_list,
):
    # total_frames = len(frame_names)
    video_segments = {}

    with tempfile.TemporaryDirectory() as cache_dir:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            base_path = os.path.join(cache_dir, "video_seg_cache")
            os.makedirs(base_path, exist_ok=True)

            # === 正向推理 ===
            forward_dir = os.path.join(base_path, "forward_video")
            os.makedirs(forward_dir, exist_ok=True)

            # 拷贝所有帧，保持原顺序
            for i, fname in enumerate(frame_names):
                dst = os.path.join(forward_dir, f"{i:05d}.jpg")
                if not os.path.exists(dst):
                    shutil.copy(fname, dst)

            inference_state = predictor.init_state(video_path=forward_dir)

            # 使用真实的 obj_id 遍历 permuted_pred_bboxs_list
            for permuted_pred_bbox in permuted_pred_bboxs_list:
                obj_id, frame_idx, bbox = permuted_pred_bbox[:]  # 真实的 obj_id

                predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=np.array(bbox),  # 使用对应的 bbox
                )

            results = [
                (a, b, c)
                for a, b, c in predictor.propagate_in_video(
                    inference_state, start_frame_idx=0
                )
            ]
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in results:
                for i, out_obj_id in enumerate(out_obj_ids):
                    if out_obj_id not in video_segments:
                        video_segments[out_obj_id] = {}

                    video_segments[out_obj_id][out_frame_idx] = (
                        (out_mask_logits[i] > 0).cpu().numpy()
                    )
            # for (
            #     out_frame_idx,
            #     out_obj_ids,
            #     out_mask_logits,
            # ) in predictor.propagate_in_video(inference_state):
            #     for i, out_obj_id in enumerate(out_obj_ids):
            #         if out_obj_id not in video_segments:
            #             video_segments[out_obj_id] = {}

            #         video_segments[out_obj_id][out_frame_idx] = (
            #             (out_mask_logits[i] > 0).cpu().numpy()
            #         )

            # # === 反向推理 ===
            # reverse_dir = os.path.join(base_path, "reverse_video")
            # os.makedirs(reverse_dir, exist_ok=True)

            # # 拷贝帧：倒序编号
            # for i in range(total_frames):
            #     src = os.path.join(video_dir, frame_names[i])
            #     dst = os.path.join(reverse_dir, f"{total_frames - 1 - i:05d}.jpg")
            #     if not os.path.exists(dst):
            #         shutil.copy(src, dst)

            # inference_state = predictor.init_state(video_path=reverse_dir)

            # # 使用真实的 obj_id 遍历 permuted_pred_bboxs_list
            # for permuted_pred_bbox in permuted_pred_bboxs_list:
            #     obj_id, original_idx, bbox = permuted_pred_bbox[:]  # 真实的 obj_id
            #     rev_idx = total_frames - 1 - original_idx  # 在倒序视频中的索引
            #     predictor.add_new_points_or_box(
            #         inference_state=inference_state,
            #         frame_idx=rev_idx,
            #         obj_id=obj_id,
            #         box=np.array(bbox),  # 使用对应的 bbox
            #     )

            # for (
            #     out_frame_idx,
            #     out_obj_ids,
            #     out_mask_logits,
            # ) in predictor.propagate_in_video(inference_state):
            #     # 反映射回来
            #     real_idx = total_frames - 1 - out_frame_idx
            #     for i, out_obj_id in enumerate(out_obj_ids):
            #         if out_obj_id not in video_segments:
            #             video_segments[out_obj_id] = {}
            #         video_segments[out_obj_id][real_idx] = (
            #             (out_mask_logits[i] > 0).cpu().numpy()
            #         )
            predictor.reset_state(inference_state)

        # packed_video = {}
        # # visualize_and_save_video(video_dir=video_dir, frame_names=frame_names, video_segments=video_segments, permuted_pred_bboxs_list=permuted_pred_bboxs_list, output_video_path=f"output_video_{process_id}.mp4")
        # for obj_id in video_segments.keys():
        #     if obj_id not in packed_video:
        #         packed_video[obj_id] = {}
        #     for frame_idx in video_segments[obj_id].keys():
        #         packed_video[obj_id][frame_idx] = np.packbits(
        #             video_segments[obj_id][frame_idx]
        #         )

        # packed_video = {
        #     "packed_video": packed_video,
        #     "shape": list(video_segments[0][0].shape),
        # }

        # response = pickle.loads(packed_video)
        # shape = response['shape']
        # packed_video = response['packed_video']

        # unpacked_video = {}
        # # visualize_and_save_video(video_dir=video_dir, frame_names=frame_names, video_segments=video_segments, permuted_pred_bboxs_list=permuted_pred_bboxs_list, output_video_path=f"output_video_{process_id}.mp4")
        # for obj_id in packed_video.keys():
        #     if obj_id not in unpacked_video:
        #         unpacked_video[obj_id] = {}
        #     for frame_idx in packed_video[obj_id].keys():
        #         unpacked_video[obj_id][frame_idx] = np.unpackbits(packed_video[obj_id][frame_idx]).reshape((shape[0], shape[1], shape[2]))

        # visualize_and_save_video(video_dir=video_dir, frame_names=frame_names, video_segments=unpacked_video, permuted_pred_bboxs_list=permuted_pred_bboxs_list, output_video_path=f"output_video_{process_id}.mp4")

    return video_segments
