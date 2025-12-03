from pathlib import Path as pth
from typing import Union

from .revos import VideoReVOSDataset


class VideoMeVISDataset(VideoReVOSDataset):
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
        if not dataset_folder.exists():
            raise ValueError(f"Dataset folder {dataset_folder} does not exist.")
        image_folder = (
            (dataset_folder / "train/JPEGImages")
            if image_folder is None
            else pth(image_folder).resolve()
        )
        expression_file = (
            (dataset_folder / "train/meta_expressions.json")
            if expression_file is None
            else pth(expression_file).resolve()
        )
        mask_file = (
            (dataset_folder / "train/mask_dict.json")
            if mask_file is None
            else pth(mask_file).resolve()
        )
        super().__init__(
            dataset_path=dataset_path,
            select_number=select_number,
            min_frames=min_frames,
            max_frames=max_frames,
            use_prompt_forcing=use_prompt_forcing,
            image_folder=image_folder,
            expression_file=expression_file,
            mask_file=mask_file,
            *args,
            **kwargs,
        )

    def get_supplementary_dataset(self, expression_file_path):
        return VideoMeVISDataset(
            dataset_path=self.dataset_folder,
            image_folder=self.image_folder,
            expression_file=expression_file_path,
            use_prompt_forcing=self.use_prompt_forcing,
            mask_file=self.mask_file,
            select_number=self.select_number,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
        )
