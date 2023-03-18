import os
from importlib.resources import path

import config


def split_dataset(file_in, test_split: int = 10, validate_split: int = 10):
    """ split one labels file into train, test, validate labels files. Creates the following files: train_labels.json, test.json, validate.json in the same folder as the source file is located

    Args:
        file_in: path to labels file
        test_split: test percentage
        validate_split: validate percentage
    """

    base_path = os.path.dirname(file_in)
    train_out_filename = 'train_labels.json'
    test_out_filename = 'test.json'
    validate_out_filename = 'validate.json'

    if (
            os.path.isfile(os.path.join(base_path, train_out_filename))
            or os.path.isfile(os.path.join(base_path, test_out_filename))
            or os.path.isfile(os.path.join(base_path, validate_out_filename))
    ):
        raise Exception('out filename already exists')

    with open(file_in) as file:
        source_data = file.readlines()

    test_lines_count = len(source_data) * test_split / 100
    validate_lines_count = len(source_data) * validate_split / 100
    test_list = []
    validate_list = []
    train_list = []

    resolution = 10
    for i in range(resolution):
        cur_line_no = int(len(source_data) / resolution) * i
        test_list.extend(source_data[cur_line_no: int(
            cur_line_no + test_lines_count / resolution)])
        validate_list.extend(source_data[
                             int(cur_line_no + test_lines_count / resolution + 1): int(cur_line_no + test_lines_count / resolution + 1 + validate_lines_count / resolution)
                             ])

    for line in source_data:
        if line not in test_list and line not in validate_list:
            train_list.append(line)

    with open(os.path.join(base_path, train_out_filename), 'w') as file:
        file.writelines(train_list)
    with open(os.path.join(base_path, test_out_filename), 'w') as file:
        file.writelines(test_list)
    with open(os.path.join(base_path, validate_out_filename), 'w') as file:
        file.writelines(validate_list)


if __name__ == '__main__':
    split_dataset(os.path.join(config.data_root, 'train_gt.json'), 15, 15)
