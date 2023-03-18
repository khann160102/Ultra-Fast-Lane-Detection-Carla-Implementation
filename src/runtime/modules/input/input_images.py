import os

import torch
import typing

from numpy import ndarray
from tqdm import tqdm

from src.runtime.utils.dataset import LaneDataset
from src.common.config.global_config import cfg, adv_cfg


def input_images(
        process_frames: typing.Callable[[torch.Tensor, typing.List[str], typing.List[ndarray]], None],
        input_file=os.path.join(cfg.data_root, cfg.test_txt),
        data_root=cfg.data_root):
    """ load images frame by frame and passes them to process_frame

    used non-basic-cfg values:

    - test_txt

    Args:
        process_frames: function taking a list of preprocessed frames, file paths and source frames
        input_file: index txt file to process
        data_root: root directory of dataset
    """
    dataset = LaneDataset(data_root, input_file, adv_cfg.img_transform)
    # i tried to replace DataLoader with my own implementation to be able to access the source frames here
    # (which DataLoader doesnt allow as it only allows tensors and strings as return value -> i would have to convert
    #  the source frames to tensors and later back, which would have a performance impact, what is exactly i was trying
    #  to prevent)
    # but everything i did decreased performance (up to 50%). Reasons are probably because DataLoader uses c code and
    # multithreading (and probably their python code is also optimized better than mine was)
    # -> use DataLoader and load images again from disk if required
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=4)
    tqdm_bar = tqdm(loader)

    for i, data in enumerate(tqdm_bar):
        imgs, names = data
        process_frames(imgs, names, tqdm_bar = tqdm_bar)
