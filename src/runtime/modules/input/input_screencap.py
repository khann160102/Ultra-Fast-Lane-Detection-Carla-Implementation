from itertools import count

import cv2
import numpy as np
import typing

from PIL import Image
from mss import mss

from src.common.config.global_config import adv_cfg, cfg


def input_screencap(process_frames: typing.Callable, mon: dict) -> None:
    """
    record from screen
    batch size is always 1

    This is was implemented to test GTA. Its a bit difficult to use. You have to manually specify the
    position and size of your target window here. If your information are wrong (out of screen) you'll get
    a cryptic exception!
    Make sure your config resolution matches your settings here.

    used non-basic cfg options: screencap_enable_image_forwarding

    Args:
        process_frames: function taking a list of preprocessed frames, file paths and source frames
        mon: position and size of recording window, eg {'top': 0, 'left': 3440, 'width': 1920, 'height': 1080}
    """

    sct = mss()
    resize = False

    for i in count():
        screenshot = sct.grab(mon)
        image = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)

        # unsqueeze: adds one dimension to tensor array (to be similar to loading multiple images)
        frame = adv_cfg.img_transform(image).unsqueeze(0)

        if cfg.screencap_enable_image_forwarding:
            image = np.array(image)
            # resize recorded frames if resolution is different from cfg.img_height / cfg.img_width
            if i == 0 and (image.shape[0] != cfg.img_height or image.shape[1] != cfg.img_width):
                resize = True
            if resize:
                image = cv2.resize(image, (cfg.img_width, cfg.img_height))

            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            process_frames(frame, None, [image])
        else:
            process_frames(frame, None, None)
