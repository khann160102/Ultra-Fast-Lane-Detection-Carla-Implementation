import json
import os
from typing import List

import config


def reduce_h_samples(file_in: str, file_out: str, keep_h_samples: List[int]):
    """ Reduces the amount of h_samples.

    Args:
        file_in: Source labels file
        file_out: target labels file
        keep_h_samples: array of h_samples to keep (eg [10, 20, 30])

    Returns:

    """
    with open(file_in) as file:
        source_data = file.readlines()

    source_h_samples = json.loads(source_data[0])['h_samples']
    start_index = None
    for i in range(len(source_h_samples)):
        if source_h_samples[i] == keep_h_samples[0]:
            start_index = i
            break

    with open(file_out, 'w') as file:
        for dataset in source_data:
            dict = json.loads(dataset)

            reduced_lanes = []
            for lane in dict['lanes']:
                reduced_lanes.append(
                    lane[start_index:start_index + len(keep_h_samples)])

            file.write(json.dumps({
                'lanes': reduced_lanes,
                'h_samples': keep_h_samples,
                'raw_file': dict['raw_file']
            }) + '\n')


if __name__ == '__main__':
    reduce_h_samples(os.path.join(config.data_root, 'train_labels.json'),
                     os.path.join(config.data_root, 'train_labels_short.json'),
                     [380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560,
                      570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710])
    reduce_h_samples(os.path.join(config.data_root, 'test.json'),
                     os.path.join(config.data_root, 'test_short.json'),
                     [380, 390, 400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560,
                      570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690, 700, 710])
