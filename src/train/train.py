from src.train.utils.common import get_logger, cp_projects, save_model
from src.train.utils.dist_utils import dist_tqdm, dist_print
from src.train.utils.factory import get_optimizer, get_scheduler, get_metric_dict, get_loss_dict
from src.common.config import global_config

import torch
import os
import datetime

from src.common.model.model import parsingNet
from src.train.data.dataloader import get_train_loader

from src.common.config.global_config import adv_cfg
from src.train.utils.metrics import update_metrics, reset_metrics


import time


def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(
        ), cls_label.long().cuda(), seg_label.long().cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out': seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.long().cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar(
                'loss/' + loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, use_aux):
    net.train()
    progress_bar = dist_tqdm(data_loader)
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)

        loss = calc_loss(loss_dict, results, logger, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name,
                                  me_op.get(), global_step=global_step)
        logger.add_scalar(
            'meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar, 'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(
                metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss='%.3f' % float(loss),
                                     data_time='%.3f' % float(
                                         t_data_1 - t_data_0),
                                     net_time='%.3f' % float(
                                         t_net_1 - t_net_0),
                                     **kwargs)
        t_data_0 = time.time()


def main():
    torch.backends.cudnn.benchmark = True

    cfg = global_config.cfg

    work_dir = adv_cfg.train_dir

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime(
        '[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18', '34', '50', '101',
                            '152', '50next', '101next', '50wide', '101wide']

    train_loader = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, cfg.use_aux, distributed,
                                    cfg.num_lanes, cfg.train_gt)
    cls_num_per_lane = adv_cfg.cls_num_per_lane

    net = parsingNet(pretrained=True, backbone=cfg.backbone,
                     cls_dim=(cfg.griding_num + 1, cls_num_per_lane, cfg.num_lanes), use_aux=cfg.use_aux).cuda()

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(
            net, device_ids=[args.local_rank])
    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    if cfg.on_train_copy_project_to_out_dir:
        cp_projects(work_dir)

    for epoch in range(resume_epoch, cfg.epoch):
        train(net, train_loader, loss_dict, optimizer,
              scheduler, logger, epoch, metric_dict, cfg.use_aux)

        save_model(net, optimizer, epoch, work_dir, distributed)
    logger.close()


if __name__ == "__main__":
    main()
