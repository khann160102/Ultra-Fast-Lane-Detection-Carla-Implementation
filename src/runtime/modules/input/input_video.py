from itertools import count

import cv2
import torch
from PIL import Image
from numpy import ndarray

from src.common.config.global_config import adv_cfg, cfg

import typing


def input_video(process_frames: typing.Callable[[torch.Tensor, typing.List[str], typing.List[ndarray]], None],
                input_file: typing.Union[str, int] = cfg.video_input_file,
                names_file: str = None
                ):
    """
    read a video file or camera stream. batch size is always 1

    used non-basic-cfg values:

    - video_input_file


    Args:
        process_frames: function taking a list of preprocessed frames, file paths and source frames
        input_file: video file (path; string) or camera index (integer)
        names_file: list with file paths to the frames of the video; if names_file and frames (jpg's) are available the input images module can also be used
    """
    if names_file:
        with open(names_file, 'r') as file:
            image_paths = file.read().splitlines()
    # else:
    #     print('no names_file specified, some functions (output modules) might not work as they require names!')

    vid = cv2.VideoCapture(input_file)
    resize = False

    # scale / resize
    if isinstance(input_file, int):  # input is camera
        vid.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.img_width)
        vid.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.img_height)
        print(
            f'input is a camera. make sure your camera is capable of your selected resolution {cfg.img_width}x{cfg.img_height}; You are probably ok if you are not getting a message that your frames will be resized')

    for i in count():
        success, image = vid.read()
        if not success:
            break

        if i == 0:
            if image.shape[0] != cfg.img_height or image.shape[1] != cfg.img_width:
                resize = True
                print(
                    'your video file does not match the specified image size -> resizing frames, will probably impact performance.')

        if resize:
            image = cv2.resize(image, (cfg.img_width, cfg.img_height))

        frame = Image.fromarray(image)
        # unsqueeze: adds one dimension to tensor array (to be similar to loading multiple images)
        frame = adv_cfg.img_transform(frame).unsqueeze(0)

        process_frames(frame, [image_paths[i]] if names_file else None, [image])


def input_camera(process_frames: typing.Callable[[torch.Tensor, typing.List[str], typing.List[ndarray]], None],
                 camera_number: int = cfg.camera_input_cam_number,
                 ):
    """ camera input wrapper for input_video()

    used non-basic-cfg values:

    - camera_input_cam_number

    Args:
        process_frames: function taking a list of preprocessed frames, file paths and source frames
        camera_number: opencv camera index
    """
    input_video(process_frames, camera_number)
