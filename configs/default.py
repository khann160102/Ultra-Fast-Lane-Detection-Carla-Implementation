"""
Default values for all config options

IMPORTANT: Changes to the values of the default config will probably brick existing configs!
"""

# BASICS
#: Basic modes - train, runtime; special modes - test, preview, prod, benchmark
from typing import List

mode: str = None
#: dataset name
dataset: str = None
#: absolute path to root directory of your dataset
data_root: str = None
#: number of samples to process in one batch
batch_size: int = 4
#: define which resnet backbone to use, allowed values - ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']
backbone: str = '18'
#: x resolution of nn, just like h_samples are the y resolution
griding_num: int = 100
#: suffix for working directory (probably good to give them a rememberable name
note: str = ''
#: absolute path to working directory: every output will be written here
work_dir: str = None
#: number of lanes
num_lanes: int = 4
#: relative height of y coordinates between 0 and 1. This is required to support different img_heights.
#: initialize this entry with something like: `[x / 720 for x in range(380, 711, 10)]`.
#:
#: to access the correct h_samples for your resolution you can use something like
#: [int(round(x*img_height)) for x in h_samples]
#:
#: see documentation for more infos
h_samples: List[float] = None
#: input image height or desired height, depending on input module. Some input modules of the runtime module (e.g. screencap) might resize their actual input res to the values specified here
img_height: int = 720
#: input image width or desired width, depending on input module. Some input modules of the runtime module (e.g. screencap) might resize their actual input res to the values specified here
img_width: int = 1280
#: resolution the neural network is working with
#:
#: untested, changing these values might not work as expected. If changed use a multiple of 8
#: some (possibly) relations in source code are unclear and might not be adjusted correctly
train_img_height: int = 288
#: resolution the neural network is working with
#:
#: untested, changing these values might not work as expected. If changed use a multiple of 8
#: some (possibly) relations in source code are unclear and might not be adjusted correctly
train_img_width: int = 800

# TRAIN
#: adding extra segmentation to improve training, read the ufld paper for more details
use_aux: bool = True
#: set via cli, required if using distributed learning (which i am not supporting, but also did not remove related code by purpose)
local_rank = None
#: number of epochs to train
epoch: int = 100
#: which optimizer to use, valid values are ['SGD','Adam']
optimizer: str = 'Adam'
#: initial learning rate
learning_rate: float = 4e-4
weight_decay: float = 1e-4
momentum: float = 0.9
scheduler: str = 'cos'  # ['multi', 'cos']
steps: List[int] = [25, 38]
gamma: float = 0.1
warmup: str = 'linear'
warmup_iters: int = 100
sim_loss_w: float = 1.0
shp_loss_w: float = 0.0
finetune = None
#: absolute path of existing model; continue training this model
resume: str = None
#: training index file
train_gt: str = 'train_gt.txt'
#: define whether the project project directory is copied to the output directory
on_train_copy_project_to_out_dir: bool = True

# RUNTIME
#: load trained model and use it for runtime
trained_model: str = None
#: specifies output module, can define multiple modules by using this parameter multiple times. Using multiple out-modules might decrease performance significantly
output_mode: List[str] = ['test']
#: specifies input module
input_mode: str = 'images'
#: enable speed measurement
measure_time: bool = False
#: testing index file
test_txt: str = 'test.txt'

# INPUT MODULES
#: full filepath to video file you want to use as input
video_input_file: str = None
#: number of your camera
camera_input_cam_number: int = 0
#: position and size of recording area: x (left), y(top), w, h
screencap_recording_area: List[int] = [0, 0, 1920, 1080]
#: allows disabling image forwarding. While this will probably improve performance for this input it will prevent you from using most out_modules as also no input_file (with paths to frames on disk) is available in this module
screencap_enable_image_forwarding: bool = True  # Disabling this will prevent you from using most out modules. Probably only usefull in some edge cases

# OUTPUT MODULES
#: file containing labels for test data to validate test results
test_validation_data: str = 'test.json'
#: enable/disable live preview
video_out_enable_live_video: bool = True
#: enable/disable export to video file
video_out_enable_video_export: bool = False
#: enable/disable export to images (like video, but as jpegs)
video_out_enable_image_export: bool = False
#: enable/disable visualize as lines instead of points
video_out_enable_line_mode: bool = False
