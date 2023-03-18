# init config, this has to be done first as its values are used in method declarations
from .train import train
from .runtime import runtime
from .common.config import global_config

global_config.init()


__all__ = ['runtime', 'train']
