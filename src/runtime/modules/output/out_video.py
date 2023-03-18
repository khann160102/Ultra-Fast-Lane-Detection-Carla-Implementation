import os
import time
from typing import Tuple, List
from PIL import Image

import cv2
import numpy as np
import torch

from src.runtime.modules.output.common import get_filename_date_string, map_x_to_image, evaluate_predictions
from src.common.config.global_config import cfg, adv_cfg


def get_lane_color(i: int) -> Tuple:
    """ Get a predefined colors depending on i. Colors repeat if i gets to big

    Args:
        i: any number, same number -> same color

    Returns: Tuple containing 3 values, eg (255, 0, 0)
    """
    lane_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                   (255, 255, 0), (255, 0, 255)]
    return lane_colors[i % 5]


class VisualOut:
    """
    provides different visual output types

    * live video
    * record video
    * save images

    visualization can be points or lines
    """

    def __init__(
            self,
            enable_live_video=cfg.video_out_enable_live_video,
            enable_video_export=cfg.video_out_enable_video_export,
            enable_image_export=cfg.video_out_enable_image_export,
            enable_line_mode=cfg.video_out_enable_line_mode,
    ):
        """
        used non-basic-cfg values:

        - cfg.video_out_enable_live_video
        - cfg.video_out_enable_video_export
        - cfg.video_out_enable_image_export
        - cfg.video_out_enable_line_mode

        Args:
            enable_live_video: show video
            enable_video_export: save as video to disk
            enable_image_export: save as image files to disk
            enable_line_mode: visualization as lines instead of dots
        """
        self.enable_live_video = enable_live_video
        self.enable_video_export = enable_video_export
        self.enable_image_export = enable_image_export
        self.enable_line_mode = enable_line_mode
        self.num = 0
        self.path = os.path.join(cfg.work_dir, '_out')
        dir_num = len(os.listdir(self.path))
        self.path = os.path.join(self.path, str(dir_num).zfill(3))
        try:
            os.mkdir(self.path)
        except:
            pass

        if enable_video_export:
            # init video out
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out_filename = f'{cfg.dataset}{cfg.note}_{os.path.basename(cfg.trained_model).split(".")[0]}_{get_filename_date_string()}.avi'
            out_full_path = os.path.join(cfg.work_dir, out_filename)
            print(out_full_path)
            self.vout = cv2.VideoWriter(
                out_full_path, fourcc, 30.0, (cfg.img_width, cfg.img_height))

    def out(self, y: torch.Tensor, names: List[str], frames: List[np.ndarray]):
        """ Generate visual output

        Args:
            y: network result (list of samples containing probabilities per sample)
            names: filenames for y, if empty: frames have to be provided
            frames: source frames, if empty: names have to be provided
        """
        if not names and not frames:
            raise Exception('at least frames or names have to be provided')
        # iterate over samples
        for i in range(len(y)):
            # get x coordinates based on probabilities
            lanes = np.array(map_x_to_image(evaluate_predictions(y[i])))

            if frames:
                vis = frames[i]
            else:
                vis = cv2.imread(os.path.join(cfg.data_root, names[i]))

            if vis is None:
                raise Exception('failed to load frame')

            for i in range(lanes.shape[0]):  # iterate over lanes
                lane = lanes[i, :]
                if np.sum(lane != -2) > 2:  # If more than two points found for this lane
                    color = get_lane_color(i)
                    for j in range(lanes.shape[1]):
                        img_x = lane[j]
                        img_y = adv_cfg.scaled_h_samples[j]
                        if img_x != -2:
                            if self.enable_line_mode:
                                # find all previous points for current lane (in reverse order) that are not -2
                                # and store indexes of these points in prev_points
                                prev_points = [x for x in range(
                                    j - 1, -1, -1) if lane[x] != -2]
                                if prev_points:
                                    cv2.line(
                                        vis,
                                        (lane[prev_points[0]],
                                         adv_cfg.scaled_h_samples[prev_points[0]]),
                                        (img_x, img_y), color,
                                        5
                                    )
                            else:
                                cv2.circle(vis, (img_x, img_y), 5, color, -1)
            if self.enable_live_video:
                self.num += 1
                img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
                img = img.save(os.path.join(
                    self.path, str(self.num).zfill(4) + '.jpg'))

                cv2.imshow('video', vis)
                cv2.waitKey(1)
            if self.enable_video_export:
                self.vout.write(vis)
            if self.enable_image_export:
                out_path = os.path.join(
                    cfg.work_dir,
                    f'{get_filename_date_string()}_out', names[i] if names else int(
                        time.time() * 1000000)
                )  # use current timestamp (nanoseconds) as fallback
                cv2.imwrite(out_path, vis)
