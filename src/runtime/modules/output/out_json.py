import json
import os
from typing import List

import numpy as np

from src.runtime.modules.output.common import get_filename_date_string, map_x_to_image, evaluate_predictions
from src.common.config.global_config import cfg, adv_cfg


class JsonOut:
    """
    provides the ability to output detected data in a json like format (one json object per line) to a file
    This file will be analog to the source labels you are using for training
    """

    def __init__(
            self,
            filepath=os.path.join(
                cfg.work_dir,
                f'{get_filename_date_string()}_{cfg.dataset}_{os.path.splitext(os.path.basename(cfg.test_txt)[-1])[0]}.json'
            )
    ):
        """
        Args:
            filepath: full file path where the results will be stored
        """
        self.filepath = filepath
        self.out_file = open(self.filepath, 'w')

    def out(self, y, names, frames: List[np.ndarray]):
        """ Generate json output to text file

        Args:
            y: network result (list of samples)
            names: filenames for y
        """
        # iterate over samples
        for i in range(len(y)):
            lanes = map_x_to_image(evaluate_predictions(y[i]))  # get x coordinates based on probabilities

            json_string = json.dumps({
                'lanes': lanes,
                'h_samples': adv_cfg.scaled_h_samples,
                'raw_file': names[i]
            })

            self.out_file.write(json_string + '\n')
