import torch
import os

import torchvision.transforms as transforms
import src.train.data.mytransforms as mytransforms
from src.train.data.dataset import LaneClsDataset
from src.common.config import global_config
from src.common.config.global_config import adv_cfg


def get_train_loader(batch_size, data_root, griding_num, use_aux, distributed, num_lanes, train_gt):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask(
            (global_config.cfg.train_img_height, global_config.cfg.train_img_width)),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((int(global_config.cfg.train_img_height / 8), int(
            global_config.cfg.train_img_width / 8))),
        mytransforms.MaskToTensor(),
    ])
    img_transform = adv_cfg.img_transform
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])
    train_dataset = LaneClsDataset(data_root,
                                   os.path.join(data_root, train_gt),
                                   img_transform=img_transform, target_transform=target_transform,
                                   simu_transform=simu_transform,
                                   segment_transform=segment_transform,
                                   row_anchor=global_config.adv_cfg.train_h_samples,
                                   # row_anchor=culane_row_anchor,
                                   griding_num=griding_num, use_aux=use_aux, num_lanes=num_lanes)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    else:
        sampler = torch.utils.data.RandomSampler(
            train_dataset)  # same as shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)

    return train_loader
