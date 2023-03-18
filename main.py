import sys
from src import runtime, train
from src.common.config.global_config import cfg


def main():
    if not cfg.data_root or not cfg.work_dir:
        raise Exception('data_root and work_dir have to be specified')

    if cfg.mode == 'runtime':
        if not cfg.trained_model:
            raise Exception('define your trained_model')
        try:
            runtime.main()
        except KeyboardInterrupt:
            print('quitting because of keyboard interrupt (probably ctrl + c)')
            sys.exit(0)
    elif cfg.mode == 'train':
        train.main()
    else:
        raise Exception('invalid mode')


if __name__ == "__main__":
    main()
