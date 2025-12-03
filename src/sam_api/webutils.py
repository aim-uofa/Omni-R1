import base64
import copy
import io
import json
import socket
import struct  # 用于处理长度前缀
import zlib

import numpy as np
import PIL
import PIL.Image

PORT = 32224
HOST = "127.0.0.1"


def get_iou(
    pred_mask: np.ndarray, gt_mask: np.ndarray, ignore_label: int = -1
) -> float:
    """
    计算推理得到的 mask 和 ground truth mask 之间的 IoU。

    Args:
        pred_mask (np.ndarray): 推理得到的 mask，形状为 (H, W)。
        gt_mask (np.ndarray): Ground truth mask，形状为 (H, W)。
        ignore_label (int): 在 gt_mask 中需要忽略的像素值。

    Returns:
        float: IoU 值，如果 union 为 0，则返回 0.0。
    """
    # 检查输入类型和形状
    if not isinstance(pred_mask, np.ndarray) or not isinstance(gt_mask, np.ndarray):
        raise ValueError("Both pred_mask and gt_mask must be NumPy arrays.")
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("pred_mask and gt_mask must have the same shape.")

    # 忽略 gt_mask 中的 ignore_label
    valid_mask = gt_mask != ignore_label

    # 计算交集和并集
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    # 避免除以零
    if union == 0:
        return 0.0

    return intersection / union


def overlay_mask(image, mask, color=(0, 255, 0), alpha=0.5):
    if mask is None:
        return image
    if len(mask.shape) == 3:
        mask = mask[0]
    overlay = np.array(image)
    binary_mask = mask
    color_mask = np.zeros_like(image)
    color_mask[binary_mask > 0] = color
    for c in range(0, 3):
        overlay[:, :, c] = np.where(
            binary_mask > 0,
            overlay[:, :, c] * (1 - alpha) + color_mask[:, :, c] * alpha,
            overlay[:, :, c],
        )
    return overlay


def send_request(data, host=HOST, port=PORT):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # 添加长度前缀
    data = data.encode("utf-8")
    data_length = struct.pack(">I", len(data))  # 使用4字节大端表示数据长度
    client_socket.sendall(data_length + data)

    # 接收响应
    response_length = struct.unpack(">I", client_socket.recv(4))[0]  # 先接收响应长度
    response = b""
    while len(response) < response_length:
        response += client_socket.recv(1024 * 1024)  # 循环接收完整响应
    client_socket.close()
    return response.decode("utf-8")


def send_request_video(data, host=HOST, port=PORT):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))

    # 添加长度前缀
    data = data.encode("utf-8")
    data_length = struct.pack(">I", len(data))  # 使用4字节大端表示数据长度
    client_socket.sendall(data_length + data)

    # 接收响应
    response_length = struct.unpack(">I", client_socket.recv(4))[0]  # 先接收响应长度
    response = b""
    while len(response) < response_length:
        response += client_socket.recv(1024 * 1024)  # 循环接收完整响应
    client_socket.close()
    return response


def request_predict_with_points(image, input_point, input_label):
    """
    发送请求到服务器进行预测
    :param image: 图像数据
    :param input_point: 输入的点坐标
    :param input_label: 输入的点标签
    :return: 服务器返回的响应
    """
    if image.__class__ == PIL.Image.Image:
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    elif image.__class__ == str:
        with open(image, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")
    request = {
        "command": "predict_with_points",
        "image": image_base64,
        "input_point": input_point.tolist(),
        "input_label": input_label.tolist(),
    }
    response = send_request(json.dumps(request))
    return json.loads(response)


def request_predict_with_box(image: PIL.Image, box):
    """
    发送请求到服务器进行预测
    :param image: 图像数据
    :param box: 输入框坐标，格式为[x1, y1, x2, y2]
    :return: 服务器返回的响应
    """
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    request = {
        "command": "predict_with_box",
        "image": image_base64,
        "input_box": box.tolist(),
    }
    response = send_request(json.dumps(request))
    return json.loads(response)


def request_predict_with_points_and_box(
    image: PIL.Image, box, input_point, input_label, mask_input=None
):
    """
    发送请求到服务器进行预测
    :param image: 图像数据
    :param box: 输入框坐标，格式为[x1, y1, x2, y2]
    :param mask_input: 输入的掩码
    :param input_point: 输入的点坐标
    :param input_label: 输入的点标签
    :return: 服务器返回的响应
    """
    image_buffer = io.BytesIO()
    image.save(image_buffer, format="PNG")
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode("utf-8")
    if mask_input is None:
        request = {
            "command": "predict_with_points_and_box",
            "image": image_base64,
            "input_box": box.tolist(),
            "input_point": input_point.tolist(),
            "input_label": input_label.tolist(),
        }
    else:
        request = {
            "command": "predict_with_points_and_box",
            "image": image_base64,
            "input_box": box.tolist(),
            "mask_input": mask_input.tolist(),
            "input_point": input_point.tolist(),
            "input_label": input_label.tolist(),
        }

    response = send_request(json.dumps(request))
    return json.loads(response)


def request_predict_with_video(
    video_dir, bbox, frame_names, mask_input=None, host=HOST, port=PORT
):
    """ """
    if bbox.__class__ == np.ndarray:
        bbox = bbox.tolist()

    request = {
        "command": "video",
        "video_dir": video_dir,
        "frame_names": frame_names,
        "pred_bbox": bbox,
    }
    # print(f'Sending bbox: {bbox}')

    response = send_request_video(json.dumps(request), host=host, port=port)

    response = zlib.decompress(response)
    # load with pickle
    import pickle

    try:
        response = pickle.loads(response)
        shape = response["shape"]
        packed_video = response["packed_video"]

        unpacked_video = {}
        # visualize_and_save_video(video_dir=video_dir, frame_names=frame_names, video_segments=video_segments, permuted_pred_bboxs_list=permuted_pred_bboxs_list, output_video_path=f"output_video_{process_id}.mp4")
        for obj_id in packed_video.keys():
            if obj_id not in unpacked_video:
                unpacked_video[obj_id] = {}
            for frame_idx in packed_video[obj_id].keys():
                unpacked_video[obj_id][frame_idx] = np.unpackbits(
                    packed_video[obj_id][frame_idx]
                ).reshape((shape[0], shape[1], shape[2]))
        return unpacked_video
    except:
        print(f"########### Error in unpickling response: {response}")
        return {}
    return response


if __name__ == "__main__":
    # predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
    import time

    start_time = time.time()
    for i in range(10):
        image = PIL.Image.open(
            "/home/zmz/code2/Seg-Zero/datasets4/GSEval/unlabeled2017/000000572774.jpg"
        ).convert("RGB")
        # image = np.array(image)
        # "box": [161.25, 144.9375, 245.75, 279.5625]
        box = np.array([161, 144, 245, 279])
        input_point = np.array([[100, 100], [150, 150]])
        input_label = np.array([1, 0])

        res = request_predict_with_box(image, box)
        # res = request_predict_with_points_and_box(image, box, input_point, input_label)
        print("Requesting with points and box...", i)
        # res = request_predict_with_box(image, input_point, input_label)

        mask = np.array(res["masks"])

        # print("Response:", res.keys())
        # predictor.set_image(image)

        # predictor.predict(
        #     box=box,
        #     point_coords=input_point,
        #     point_labels=input_label,
        #     multimask_output=False
        # )
        import torchshow

        torchshow.save(mask, title="mask", cmap="gray")
        overlay_img = overlay_mask(
            copy.deepcopy(image), mask, color=(0, 255, 0), alpha=0.5
        )
        torchshow.save(overlay_img, title="overlay_mask", cmap="gray")
    end_time = time.time()

    print("Time taken for 100 requests:", end_time - start_time)

    # 示例请求
    # with open('/mnt/petrelfs/liumingyu/code/zzcode/Omini-R1-omini_zz/overlay.jpg', 'rb') as f:
    #     image_base64 = base64.b64encode(f.read()).decode('utf-8')

    # request = {
    #     'command': 'predict_with_box',
    #     'image': image_base64,
    #     'input_box': [50, 50, 200, 200]
    # }
    # # print(request)
    # response = send_request(json.dumps(request))
    # response_json = json.loads(response)
    # print(response_json.keys())
    # print("Response:", )
