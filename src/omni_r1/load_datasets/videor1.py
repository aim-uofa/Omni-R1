import os
from pathlib import Path as pth

import torch
from datasets import Dataset, load_from_disk
from rouge_score import rouge_scorer

QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> </think> tags, and then give your final answer between the <answer> </answer> tags."
)

TYPE_TEMPLATE = {
    "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
    "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
    "free-form": " Please provide your text answer within the <answer> </answer> tags.",
    "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
}


class VideoR1Dataset(Dataset):
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
            raise ValueError(f"Video-R1 Dataset path {dataset_path} does not exist.")
        self.dataset_path = dataset_path
        self.dataset = Dataset.from_json(str(dataset_path))
        # self.rank0_load(batched= batch_size > 1, batch_size=batch_size, num_proc=num_proc)

    def rank0_load(self, **kwargs):
        # 获取分布式环境信息
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        cache_path = "/dev/shm/processed_dataset_cache"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # 仅在主进程执行预处理,其他进程等待
        if local_rank == 0:
            self.process_data(**kwargs)
            # 保存到共享位置
            # cache_path = "./tmp/processed_dataset_cache"
            self.dataset.save_to_disk(cache_path)

        # 同步所有进程
        if world_size > 1:
            torch.distributed.barrier()

        # 所有进程从缓存加载
        if local_rank != 0:
            self.dataset = load_from_disk(cache_path)

    def process_data(self, **kwargs):
        def make_conversation_image_and_video(examples):
            prompts = []

            for i in range(len(examples["problem"])):
                if examples["problem_type"][i] == "multiple choice":
                    question = examples["problem"][i] + "Options:\n"
                    for op in examples["options"][i]:
                        question += op + "\n"
                else:
                    question = examples["problem"][i]

                msg = {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": examples["data_type"][i],
                                    examples["data_type"][i]: str(
                                        self.dataset_path.parent
                                        / str(examples["path"][i][2:])
                                    ),
                                },
                                {
                                    "type": "text",
                                    "text": QUESTION_TEMPLATE.format(Question=question)
                                    + TYPE_TEMPLATE[examples["problem_type"][i]],
                                },
                            ],
                        }
                    ]
                }
                prompts.append(msg)

            return {"prompt": [p["prompt"] for p in prompts]}

        self.dataset = self.dataset.map(make_conversation_image_and_video, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """在线处理数据"""

        items = self.dataset[idx]

        # 构建问题
        if items["problem_type"] == "multiple choice":
            question = items["problem"] + "Options:\n"
            for op in items["options"]:
                question += op + "\n"
        else:
            question = items["problem"]

        items["path"] = str(self.dataset_path.parent / str(items["path"][2:]))

        # 构建提示
        prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": items["data_type"],
                        items["data_type"]: items[
                            "path"
                        ],  # str(self.dataset_path.parent / str(items["path"][i][2:])),
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=question)
                        + TYPE_TEMPLATE[items["problem_type"]],
                    },
                ],
            }
        ]

        items["prompt"] = prompt
        items["is_video_r1"] = True

        # 返回处理后的数据
        return items


def video_r1_reward(
    question_type: str, output_ans: str, gt_ans: str, **kwargs
) -> float:
    def normalize_number(num_str):
        try:
            num_str = num_str.replace(",", "")
            return float(num_str)
        except Exception as e:
            print(f"Error converting '{num_str}' to float: {e}")
            return None

    def wer(reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        m = len(ref_words)
        n = len(hyp_words)
        d = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = 1 + min(d[i - 1][j], d[i][j - 1], d[i - 1][j - 1])
        return d[m][n] / max(1, m)

    def compute_rouge_score(reference, hypothesis, use_stemmer=True):
        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=use_stemmer
        )
        scores = scorer.score(reference, hypothesis)
        average_fmeasure = (
            scores["rouge1"].fmeasure
            + scores["rouge2"].fmeasure
            + scores["rougeL"].fmeasure
        ) / 3
        return average_fmeasure

    if question_type == "multiple choice":
        reward = 1.0 if output_ans.strip() == gt_ans.strip() else 0.0

    elif question_type == "numerical":
        gt_has_decimal = ("." in gt_ans) or ("," in gt_ans)
        out_has_decimal = ("." in output_ans) or ("," in output_ans)
        if gt_has_decimal != out_has_decimal:
            reward = 0.0
        else:
            gt_number = normalize_number(gt_ans)
            out_number = normalize_number(output_ans)
            if gt_number is None or out_number is None:
                reward = 0.0
            else:
                reward = 1.0 if round(gt_number, 2) == round(out_number, 2) else 0.0

    elif question_type == "OCR":
        error_rate = wer(gt_ans, output_ans)
        reward = 1 - error_rate
        reward = max(0.0, min(1.0, reward))

    elif question_type == "free-form":
        score = compute_rouge_score(gt_ans, output_ans)
        reward = max(0.0, min(1.0, score))

    elif question_type == "regression":
        gt_number = normalize_number(gt_ans)
        out_number = normalize_number(output_ans)
        if gt_number is None or out_number is None:
            reward = 0.0
        rel_diff = (abs(out_number - gt_number) + 1e-9) / (abs(gt_number) + 1e-9)
        rel_diff = min(1.0, max(0.0, rel_diff))
        reward = 1 - rel_diff
