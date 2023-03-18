import datetime
import os
import sys
from typing import Any

from torchvision import transforms

from src.common.config.cli_parser import get_args
from src.common.config.util import Config

cfg = None
adv_cfg = None


def __set_config(key: str, value: Any):
    """ Update config entry. You should not use this function in your code, once the cfg is set up it should not be changed again!

    Args:
        key: name of config entry
        value: new value
    """
    # print('update ', (key, value), ' config')
    setattr(cfg, key, value)


def process_modes():
    """
    handling of special runtime modes
    """
    if not cfg.mode:
        raise Exception('select mode')

    if cfg.mode in ['test', 'preview', 'prod', 'benchmark']:
        print(
            f'Special mode {cfg.mode} selected, this will override some of your config settings.')
        if cfg.mode == 'test':
            if not cfg.test_txt or not cfg.test_validation_data:
                raise Exception(
                    'test_txt and test_validation_data has to be specified for testing mode')
            __set_config('input_mode', 'images')
            __set_config('output_mode', ['test'])
        elif cfg.mode == 'preview':
            if not cfg.test_txt:
                raise Exception(
                    'test_txt has to be specified for testing mode')
            __set_config('input_mode', 'images')
            __set_config('output_mode', ['video'])
            __set_config('video_out_enable_live_video', True)
            __set_config('batch_size', 1)
        elif cfg.mode == 'prod':
            __set_config('input_mode', 'camera')
            __set_config('output_mode', ['prod'])
        elif cfg.mode == 'benchmark':
            __set_config('input_mode', 'images')
            __set_config('output_mode', ['json'])
            __set_config('measure_time', True)

        cfg.mode = 'runtime'


def merge_config() -> Config:
    """
    combines default and user-config and cli arguments
    """
    args = get_args().parse_args()
    user_cfg = Config.fromfile(args.config)
    cfg = Config.fromfile('configs/default.py')

    # override default cfg with values from user cfg
    for item in [(k, v) for k, v in user_cfg.items() if v]:
        cfg[item[0]] = item[1]

    for k, v in vars(args).items():
        if v is not None:
            # print('update ', (k, v), ' config')
            setattr(cfg, k, v)
    return cfg


class AdvancedConfig:
    """
    This class provides "advanced config values" meaning it calculates some values which depend on other cfg values
    They are calculated here to

    - keep the configs clean
    - prevent calculating them multiple times during runtime which should make your code cleaner and improve performance
    """

    @staticmethod
    def gen_train_dir(cfg):
        now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        hyper_param_str = '_lr_%1.0e_b_%d' % (
            cfg.learning_rate, cfg.batch_size)
        work_dir = os.path.join(cfg.work_dir, now + hyper_param_str + cfg.note)
        return work_dir

    def __init__(self):
        # import pdb;pdb.set_trace()
        self.cls_num_per_lane = len(cfg.h_samples)
        self.scaled_h_samples = [int(round(x * cfg.img_height))
                                 for x in cfg.h_samples]
        self.train_h_samples = [
            x * cfg.train_img_height for x in cfg.h_samples]
        self.img_transform = transforms.Compose([
            transforms.Resize((cfg.train_img_height, cfg.train_img_width)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.train_dir = self.gen_train_dir(cfg)


def init():
    """
    in the good old times this was a simple import and forget, but this behaviour breaks sphinx :/
    Now the config has to be initialized once at start
    """
    global cfg, adv_cfg
    if not cfg:
        cfg = merge_config()
        process_modes()
        adv_cfg = AdvancedConfig()


class Dummy:
    """
    This is a simple dummy class (mock) which will always return a dummy string for every value its asked for
    This prevents errors if this applications is run under unusual circumstances (eg doc generation)
    """

    def __init__(self, name):
        """
        Args:
            name: Name of the class/object this instantiation will replace. eg 'cfg'
        """
        self.name = name

    def __getattribute__(self, item):
        return f"{object.__getattribute__(self, 'name')}.{item}"


# Use mock class if this file is called during doc generation
if 'sphinx' in sys.modules and not cfg:
    cfg = Dummy('cfg')
    adv_cfg = Dummy('adv_cfg')
