import argparse
import base64
import copy
import io
import json
import multiprocessing
import os
import pickle
import queue
import shutil
import signal
import socket
import struct
import tempfile
import time
import zlib

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

multiprocessing.set_start_method("spawn", force=True)

SAM_TYPE = torch.bfloat16
USE_LOCAL = True
VOS_OPTIMIZED = False  # Whether to use VOS optimized predictor


# 定义工作函数
def worker_process(
    task_queue, result_queue, worker_id, use_image=False, use_video=True
):
    """工作进程函数，用于处理预测任务"""
    print(f"Worker {worker_id} started")

    gpu_id = worker_id % torch.cuda.device_count()

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")

    # 在工作进程中初始化预测器
    predictors = {}
    if USE_LOCAL:
        if use_image:
            predictors["image"] = build_sam2_video_predictor(
                config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                ckpt_path="sam2/sam2.1_hiera_large.pt",
                torch_dtype=SAM_TYPE,
                device=device,
            )
        # FIXME: vos_optimized
        if use_video:
            predictors["video"] = build_sam2_video_predictor(
                config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
                ckpt_path="sam2/sam2.1_hiera_large.pt",
                torch_dtype=SAM_TYPE,
                device=device,
                vos_optimized=VOS_OPTIMIZED,
            )
    else:
        if use_image:
            predictors["image"] = SAM2ImagePredictor.from_pretrained(
                "facebook/sam2-hiera-large", torch_dtype=SAM_TYPE, device=device
            )
        if use_video:
            predictors["video"] = SAM2VideoPredictor.from_pretrained(
                "facebook/sam2-hiera-large", torch_dtype=SAM_TYPE, device=device
            )
    print(f"Worker {worker_id} initialized.")
    # 设置信号处理
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        try:
            # 从任务队列获取任务
            client_id, data = task_queue.get()
            if client_id is None:  # 终止信号
                break

            # 处理请求
            response = handle_request(data, predictors, worker_id)

            # 将结果放入结果队列
            result_queue.put((client_id, response))

        except Exception as e:
            print(f"Worker {worker_id} error: {str(e)}")
            # 在出错的情况下也要返回错误信息
            if "client_id" in locals():
                error_response = json.dumps({"error": str(e)})
                result_queue.put((client_id, error_response))

    print(f"Worker {worker_id} stopped")


def predict_with_box(image: np.array, input_box: np.array, predictor):
    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    return masks, scores, logits


def predict_with_points_and_box(
    image: np.array,
    input_box: np.array,
    input_point: np.array,
    input_label: np.array,
    predictor,
    mask_input: np.array = None,
):
    predictor.set_image(image)
    if mask_input is not None:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
            mask_input=mask_input[None, :, :],
            multimask_output=False,
        )
    else:
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box[None, :],
            multimask_output=False,
        )
    return masks, scores, logits


def predict_with_points(
    image: np.array, input_point: np.array, input_label: np.array, predictor
):
    predictor.set_image(image)
    box = np.array([0, 0, image.shape[0], image.shape[1]])
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
        box=box[None, :],
    )
    return masks, scores, logits


def visualize_and_save_video(
    video_dir,
    frame_names,
    video_segments,
    permuted_pred_bboxs_list,
    output_video_path,
    fps=2,
):
    """可视化分割结果并保存为视频"""
    # 获取第一帧的大小
    first_frame = cv2.imread(os.path.join(video_dir, frame_names[0]))
    height, width, _ = first_frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_idx, fname in enumerate(frame_names):
        # 读取当前帧
        frame = cv2.imread(os.path.join(video_dir, fname))

        # 为当前帧绘制所有目标的 bbox 和 mask
        for permuted_pred_bbox in permuted_pred_bboxs_list:
            if len(permuted_pred_bbox) != 3:
                raise ValueError(
                    f"Expected permuted_pred_bbox to have 3 elements, but got {len(permuted_pred_bbox)}: {permuted_pred_bbox}"
                )

            obj_id, frame_idx_pred, bbox = permuted_pred_bbox[:]

            if frame_idx == frame_idx_pred:
                # 绘制边界框
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 绘制obj_id
                label = f"obj_id: {obj_id}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )

            # 绘制分割掩码
            for obj_id in video_segments.keys():
                mask = video_segments[obj_id][frame_idx]

                # 将掩码调整为视频帧大小
                mask_resized = cv2.resize(
                    mask[0].astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )

                # 创建一个全零的图像，与 frame 大小一致
                mask_colored = np.zeros_like(frame)
                mask_colored[mask_resized == 1] = [0, 0, 255]  # 红色标记掩码区域

                # 合并原始帧和掩码
                frame = cv2.addWeighted(frame, 1, mask_colored, 0.5, 0)

        # 将处理过的帧写入视频文件
        out_video.write(frame)

    # 释放视频写入对象
    out_video.release()
    print(f"视频已保存至: {output_video_path}")


def seg_bidirectional_from_bbox(
    predictor,
    video_dir,
    frame_names,
    permuted_pred_bboxs_list,
    process_id=0,
    cache_dir="./.seg_cache",
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
                    # shutil.copy(os.path.join(video_dir, fname), dst)

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

        packed_video = {}
        # visualize_and_save_video(video_dir=video_dir, frame_names=frame_names, video_segments=video_segments, permuted_pred_bboxs_list=permuted_pred_bboxs_list, output_video_path=f"output_video_{process_id}.mp4")
        for obj_id in video_segments.keys():
            if obj_id not in packed_video:
                packed_video[obj_id] = {}
            for frame_idx in video_segments[obj_id].keys():
                packed_video[obj_id][frame_idx] = np.packbits(
                    video_segments[obj_id][frame_idx]
                )

        packed_video = {
            "packed_video": packed_video,
            "shape": list(video_segments[0][0].shape),
        }
        packed_video = pickle.dumps(packed_video)
        sending_video = copy.deepcopy(packed_video)

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

    return sending_video


def handle_request(data, predictors, worker_id):
    """处理客户端请求"""
    try:
        request = json.loads(data)
        command = request.get("command")

        if command == "video":
            video_dir = request.get("video_dir")
            frame_names = request.get("frame_names")
            pred_bbox = request.get("pred_bbox")

            video_predictor = predictors.get("video")
            if not video_predictor:
                return json.dumps({"error": "Video predictor not initialized"})
            try:
                video_segments = seg_bidirectional_from_bbox(
                    video_predictor, video_dir, frame_names, pred_bbox, worker_id
                )
            except Exception as e:
                print(f"Error during video segmentation: {str(e)}")
                video_segments = pickle.dumps({})

            compressed_segments = zlib.compress(video_segments)

            return compressed_segments

        image_base64 = request.get("image")
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image = np.array(image)

        image_predictor = predictors.get("image")
        if not image_predictor:
            return json.dumps({"error": "Image predictor not initialized"})

        if command == "predict_with_box":
            input_box = np.array(request.get("input_box"))
            masks, scores, logits = predict_with_box(image, input_box, image_predictor)
        elif command == "predict_with_points":
            input_point = np.array(request.get("input_point"))
            input_label = np.array(request.get("input_label"))
            masks, scores, logits = predict_with_points(
                image, input_point, input_label, image_predictor
            )
        elif command == "predict_with_points_and_box":
            lists = ["input_box", "input_point", "input_label", "mask_input"]
            values = {}
            for i in range(len(lists)):
                try:
                    step_value = request.get(lists[i])
                    if step_value:
                        values[lists[i]] = np.array(step_value)
                    else:
                        values[lists[i]] = None
                except Exception as e:
                    values[lists[i]] = None
                    print(f"Error converting {lists[i]} to numpy array: {e}")

            masks, scores, logits = predict_with_points_and_box(
                image,
                values["input_box"],
                values["input_point"],
                values["input_label"],
                image_predictor,
                values["mask_input"],
            )
        else:
            print(f"Invalid command: {command}")
            return json.dumps({"error": "Invalid command"})

        response = {
            "masks": [mask.tolist() for mask in masks],
            "scores": scores.tolist(),
            "logits": [logit.tolist() for logit in logits],
        }
        return json.dumps(response)
    except Exception as e:
        import traceback

        print(f"Error handling request: {str(e)}")
        print(traceback.format_exc())
        return json.dumps({"error": str(e)})


def receive_full_data(client_socket):
    """从客户端套接字接收完整数据"""
    # 先接收长度前缀
    length_data = client_socket.recv(4)
    if not length_data:
        return None

    data_length = struct.unpack(">I", length_data)[0]  # 4字节大端表示数据长度
    data = b""
    while len(data) < data_length:
        chunk = client_socket.recv(min(1024 * 1024, data_length - len(data)))
        if not chunk:
            raise ConnectionError("Connection closed before receiving complete data")
        data += chunk
    return data.decode("utf-8")


def client_handler(
    client_socket, client_id, task_queue, result_dict, client_done_event
):
    """处理客户端连接的函数"""
    try:
        data = receive_full_data(client_socket)
        if not data:
            return

        # 将请求放入任务队列
        task_queue.put((client_id, data))

        # 等待结果，使用轮询方式检查结果字典
        max_wait_time = 600  # 最长等待时间（秒）
        wait_interval = 0.1  # 轮询间隔（秒）
        waited_time = 0

        while waited_time < max_wait_time:
            # 检查结果是否已经准备好
            if client_id in result_dict:
                # 获取结果并发送回客户端
                response = result_dict[client_id]
                # 通知已经获取到结果
                client_done_event.put(client_id)

                # 添加长度前缀到响应
                if not isinstance(response, bytes):
                    response = response.encode("utf-8")

                response_length = struct.pack(">I", len(response))
                client_socket.sendall(response_length + response)
                return

            # 轮询等待
            time.sleep(wait_interval)
            waited_time += wait_interval

        # 超时处理
        error_response = json.dumps({"error": "处理请求超时"})
        print(error_response)
        error_response = error_response.encode("utf-8")
        error_response_length = struct.pack(">I", len(error_response))
        client_socket.sendall(error_response_length + error_response)

    except Exception as e:
        print(f"Error in client handler: {str(e)}")
        try:
            error_response = json.dumps({"error": str(e)})
            error_response = error_response.encode("utf-8")
            error_response_length = struct.pack(">I", len(error_response))
            client_socket.sendall(error_response_length + error_response)
        except:
            pass
    finally:
        client_socket.close()


def result_collector(result_queue, result_dict, client_done_event):
    """从结果队列收集结果并存储到结果字典中"""
    while True:
        try:
            # 从结果队列获取结果
            client_id, result = result_queue.get()
            if client_id is None:  # 终止信号
                break

            # 将结果存储到共享字典
            result_dict[client_id] = result

        except Exception as e:
            import traceback

            print(f"结果收集器出错: {str(e)}")
            print(traceback.format_exc())

    print("结果收集器已停止")


def cleanup_resources(result_dict, client_done_event):
    """定期清理已完成的客户端资源"""
    while True:
        try:
            # 检查是否有客户端已完成处理
            try:
                # 非阻塞方式获取已完成的客户端ID
                while True:
                    client_id = client_done_event.get_nowait()
                    # 安全移除结果
                    if client_id in result_dict:
                        del result_dict[client_id]
                        print(f"已清理客户端 {client_id} 的资源")
            except queue.Empty:
                # 队列为空，继续等待
                pass

            # 周期性清理检查
            time.sleep(30)  # 每30秒检查一次

        except Exception as e:
            print(f"资源清理出错: {e}")


def start_server(host="0.0.0.0", port=32224, num_processes=4):
    """启动SAM服务器，使用多进程模型处理请求"""
    # 使用Manager管理共享对象
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    client_done_event = multiprocessing.Queue()  # 用于通知客户端已完成处理

    # 启动工作进程
    workers = []
    for i in range(num_processes):
        p = multiprocessing.Process(
            target=worker_process, args=(task_queue, result_queue, i), daemon=True
        )
        p.start()
        workers.append(p)

    # 启动结果收集器进程
    collector_thread = multiprocessing.Process(
        target=result_collector,
        args=(result_queue, result_dict, client_done_event),
        daemon=True,
    )
    collector_thread.start()

    # 启动资源清理进程
    cleaner = multiprocessing.Process(
        target=cleanup_resources, args=(result_dict, client_done_event), daemon=True
    )
    cleaner.start()

    # 设置服务器socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(10)
    print(f"Server listening on {host}:{port} with {num_processes} worker processes")

    next_client_id = 0
    client_handlers = []

    try:
        while True:
            try:
                client_socket, addr = server_socket.accept()
                print(f"Connection from {addr} (client_id: {next_client_id})")

                # 为每个客户端创建一个处理进程
                handler = multiprocessing.Process(
                    target=client_handler,
                    args=(
                        client_socket,
                        next_client_id,
                        task_queue,
                        result_dict,
                        client_done_event,
                    ),
                    daemon=True,
                )
                handler.start()
                client_handlers.append(handler)

                # 清理已完成的处理进程
                client_handlers = [h for h in client_handlers if h.is_alive()]

                next_client_id += 1
            except KeyboardInterrupt:
                print("Keyboard interrupt received, shutting down...")
                break

    except Exception as e:
        print(f"Server error: {str(e)}")
    finally:
        print("Shutting down server...")
        # 关闭所有连接并清理资源
        server_socket.close()

        # 发送终止信号给所有工作进程
        for _ in range(len(workers)):
            task_queue.put((None, None))

        # 发送终止信号给结果收集器
        result_queue.put((None, None))

        # 等待所有进程结束
        for p in workers:
            p.join(timeout=5)

        collector_thread.join(timeout=5)
        cleaner.join(timeout=5)

        print("Server shutdown complete.")


def create_parser():
    parser = argparse.ArgumentParser(description="Start SAM Server")
    parser.add_argument(
        "--port", "-p", type=int, default=32224, help="Server Port (Default: 32224)"
    )
    parser.add_argument(
        "--processes", "-n", type=int, default=4, help="instance number (Default: 4)"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Server Host (Default: 0.0.0.0)"
    )
    parser.add_argument(
        "--use_image",
        action="store_true",
        help="Instantiate Image Predictor (Default: False)",
    )
    parser.add_argument(
        "--use_video",
        action="store_true",
        help="Instantiate Video Predictor (Default: True)",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    start_server(port=args.port, num_processes=args.processes)
