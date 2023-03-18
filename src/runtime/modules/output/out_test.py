import json
import os
from typing import List

import torch

from src.runtime.modules.output.common import map_x_to_image, evaluate_predictions
from src.common.config.global_config import cfg, adv_cfg


class TestOut:
    """
    This module allows to validate predictions against known labels. It prints the accuracy after the test completed.
    Additionally it writes its results as csv to the directory where the trained model is located.
    """
    def __init__(self, out_file: str = cfg.test_validation_data):
        """
        used non-basic-cfg values:

        - cfg.test_validation_data

        Args:
            out_file: relative path to cfg.data_root
        """
        self.compare_file = os.path.join(cfg.data_root, out_file)
        self.lanes_pred = []

        try:
            from src.runtime.utils.evaluation.lane import LaneEval
            self.LaneEval = LaneEval
        except:
            print('Failed to import Evaluation code. Its either missing or something went wrong while adding it. See documentation "howto/testing/Add testing code" on how to add the required code')
            exit(1)

    def out(self, predictions: torch.Tensor, names: List[str], _):
        """ collect results of batch

        Args:
            predictions: network result (list of samples containing probabilities per sample)
            names: filenames for predictions, if empty

        """
        if not names:
            raise Exception('test output module requires "names", can\'t continue. You probably either selected the wrong in or out module.')

        for i in range(len(predictions)):
            # get x coordinates based on probabilities
            lanes = map_x_to_image(evaluate_predictions(predictions[i]))
            self.lanes_pred.append({
                'lanes': lanes,
                'h_samples': adv_cfg.scaled_h_samples,
                'raw_file': names[i]
            })

    def post(self):
        """
        Evaluate collected data and print accuracy
        """
        try:
            lanes_comp = [json.loads(line) for line in open(self.compare_file, 'r').readlines()]
        except:
            raise Exception('failed to load file with validation data')

        # some basic validation
        if len(self.lanes_pred) != len(lanes_comp):
            raise Exception('length of predicted data does not match compare data')

        res = self.LaneEval.bench_one_submit(self.lanes_pred, lanes_comp)
        res = json.loads(res)

        with open(os.path.join(os.path.dirname(cfg.trained_model), 'test_results.csv'), 'a') as f:
            line = [self.compare_file, os.path.basename(cfg.trained_model)]
            for r in res:
                print(r['name'], r['value'])
                line.append(str(r['value']))
            f.write(";".join(line) + '\n')
