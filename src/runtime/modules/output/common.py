"""
Contains some common helper functions for output modules
"""


import datetime

import numpy as np
from scipy import special

from src.common.config.global_config import cfg


def map_x_to_image(y):
    """
    Map x-axis (griding_num) estimations to image coordinates

    Args:
        y: one result sample (can be directly from net or post-processed -> all number types should be accepted)

    Returns: x coordinates for each lane
    """
    lanes = []
    offset = 0.5  # different values used in ufld project. demo: 0.0, test: 0.5

    for i in range(y.shape[1]):
        out_i = y[:, i]
        lane = [
            int((loc + offset) * float(cfg.img_width) / (cfg.griding_num - 1))
            # int(round((loc + 0.5) * float(cfg.img_width) / (cfg.griding_num - 1)))
            if loc != -2
            else -2
            for loc in out_i
        ]
        lanes.append(lane)
    return lanes


def evaluate_predictions(y):
    """
    Evaluate predictions
    Tries to improve the estimation by including all probabilities instead of only using the most probable class
    Args:
        y: one result sample

    Returns:
        2D array containing x values (float) per h_sample and lane
    """
    out = y.data.cpu().numpy()  # load data to cpu and convert to numpy
    # get most probably x-class per lane and h_sample
    out_loc = np.argmax(out, axis=0)

    # do some stuff i dont fully understand to improve x accuracy
    # relative probability with sum() == 1.0
    prob = special.softmax(out[:-1, :, :], axis=0)
    # init 3 dim array containing numbers from 0 to griding_num - 1
    idx = np.arange(cfg.griding_num).reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)  # calculate more accurate x values

    # where the most probable class is 100 (no lane detected): replace with -2
    loc[out_loc == cfg.griding_num] = -2
    return loc


def get_filename_date_string():
    """
    get current date and time in a format suitable for file exports
    Returns: string
    """
    return datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
