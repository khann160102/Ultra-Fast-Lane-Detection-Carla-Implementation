import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, argument_default=None)
    parser.usage = """python main.py <config file> [params]\nrun "python main.py --help" for more information"""
    parser.description = """Most config values can be overwritten with its corresponding cli parameters. For further details on the config options see configs/default.py and the documentation.
unavailable options on cli:
- h_samples
"""

    # @formatter:off
    # define config groups, only relevant for --help
    basics = parser.add_argument_group(
        'basic switches, these are always needed')
    train_args = parser.add_argument_group(
        'training', 'these switches are only used for training')
    runtime_args = parser.add_argument_group(
        'runtime', 'these switches are only used in the runtime module')
    in_modules = parser.add_argument_group(
        'input modules', 'with these options you can configure the input modules. Each module may have its own config switches')
    out_modules = parser.add_argument_group(
        'output modules', 'with these options you can configure the output modules. Each module may have its own config switches')

    # define switches
    basics.add_argument('config', help='path to config file')
    basics.add_argument('--mode', metavar='', type=str,
                        help='Basic modes: train, runtime; special modes: test, preview, prod, benchmark')
    basics.add_argument('--dataset', metavar='', type=str,
                        help='dataset name, can be any string')
    basics.add_argument('--data_root', metavar='', type=str,
                        help='absolute path to root directory of your dataset')
    basics.add_argument('--batch_size', metavar='', type=int,
                        help='number of samples to process in one batch')
    basics.add_argument('--backbone', metavar='', type=str,
                        help="define which resnet backbone to use, allowed values: ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']")
    basics.add_argument('--griding_num', metavar='', type=int,
                        help='x resolution of nn, just like h_samples are the y resolution')
    basics.add_argument('--note', metavar='', type=str,
                        help='suffix for working directory (probably good to give them a rememberable name')
    basics.add_argument('--work_dir', metavar='', type=str,
                        help='working directory: every output will be written here')
    basics.add_argument('--num_lanes', metavar='',
                        type=int, help='number of lanes')
    basics.add_argument('--img_height', metavar='',
                        type=int, help='height of input images')
    basics.add_argument('--img_width', metavar='', type=int,
                        help='width of input images')
    basics.add_argument('--train_img_height', metavar='', type=int,
                        help='height of image which will be passed to nn; !this option is untested and might not work!')
    basics.add_argument('--train_img_width', metavar='', type=int,
                        help='width of image which will be passed to nn; !this option is untested and might not work!')

    train_args.add_argument('--use_aux', metavar='', type=str2bool,
                            help='used to improve training, should be disabled during runtime (independent of this config)')
    train_args.add_argument('--local_rank', metavar='', type=int, default=0)
    train_args.add_argument('--epoch', metavar='',
                            type=int, help='number of epochs to train')
    train_args.add_argument('--optimizer', metavar='', type=str)
    train_args.add_argument('--learning_rate', metavar='', type=float)
    train_args.add_argument('--weight_decay', metavar='', type=float)
    train_args.add_argument('--momentum', metavar='', type=float)
    train_args.add_argument('--scheduler', metavar='', type=str)
    train_args.add_argument('--steps', metavar='', type=int, nargs='+')
    train_args.add_argument('--gamma', metavar='', type=float)
    train_args.add_argument('--warmup', metavar='', type=str)
    train_args.add_argument('--warmup_iters', metavar='', type=int)
    train_args.add_argument('--sim_loss_w', metavar='', type=float)
    train_args.add_argument('--shp_loss_w', metavar='', type=float)
    train_args.add_argument('--finetune', metavar='', type=str)
    train_args.add_argument('--resume', metavar='', type=str,
                            help='path of existing model; continue training this model')
    train_args.add_argument('--train_gt', metavar='',
                            type=str, help='training index file (train_gt.txt)')
    train_args.add_argument('--on_train_copy_project_to_out_dir', metavar='', type=str2bool,
                            help='define whether the project project directory is copied to the output directory')

    runtime_args.add_argument('--trained_model', metavar='',
                              type=str, help='load trained model and use it for runtime')
    runtime_args.add_argument('--output_mode', metavar='', type=str, action='append',
                              help='specifies output module, can define multiple modules by using this parameter multiple times. Using multiple out-modules might decrease performance significantly')
    runtime_args.add_argument(
        '--input_mode', metavar='', type=str, help='specifies input module')
    runtime_args.add_argument(
        '--measure_time', metavar='', type=str2bool, help='enable speed measurement')
    runtime_args.add_argument('--test_txt', metavar='',
                              type=str, help='testing index file (test.txt)')

    in_modules.add_argument('--video_input_file', metavar='', type=str,
                            help='full filepath to video file you want to use as input')
    in_modules.add_argument('--camera_input_cam_number',
                            metavar='', type=int, help='number of your camera')
    in_modules.add_argument('--screencap_recording_area', metavar='', type=int,
                            nargs='+', help='position and size of recording area: x,y,w,h')
    in_modules.add_argument('--screencap_enable_image_forwarding', metavar='', type=str2bool,
                            help='allows disabling image forwarding. While this will probably improve performance for this input it will prevent you from using most out_modules as also no input_file (with paths to frames on disk) is available in this module')

    out_modules.add_argument('--test_validation_data', metavar='', type=str,
                             help='file containing labels for test data to validate test results')
    out_modules.add_argument('--video_out_enable_live_video',
                             metavar='', type=str2bool, help='enable/disable live preview')
    out_modules.add_argument('--video_out_enable_video_export', metavar='',
                             type=str2bool, help='enable/disable export to video file')
    out_modules.add_argument('--video_out_enable_image_export', metavar='', type=str2bool,
                             help='enable/disable export to images (like video, but as jpegs)')
    out_modules.add_argument('--video_out_enable_line_mode', metavar='',
                             type=str2bool, help='enable/disable visualize as lines instead of points')
    # @formatter:on
    return parser
