import os

import config


def create_smaller_train_set(file_in, keep: int = 10, out_filename='small_train_labels.json'):
    """
    Training with large datasets will take a huge amount of time. Especially if the dataset was recorded at a high
    framerate this script could be used to increase training speed by creating a smaller train_labels file.

    Args:
        file_in: absolute path to a train_labels.json file
        keep: Percentage of frames to keep (20% -> 20)
        out_filename: Filename of the new labels file. It will be created in the same directory where file_in is located

    Returns:

    """
    base_path = os.path.dirname(file_in)

    if (
            os.path.isfile(os.path.join(base_path, out_filename))
    ):
        raise Exception('out filename already exists')

    with open(file_in) as file:
        source_data = file.readlines()

    keep_data = []
    for i in range(0, len(source_data), int(100/keep)):
        keep_data.append(source_data[i])

    with open(os.path.join(base_path, out_filename), 'w') as file:
        file.writelines(keep_data)


if __name__ == '__main__':
    create_smaller_train_set(os.path.join(
        config.data_root, 'train_labels.json'), int(100/5), 'small_train_labels.json')
