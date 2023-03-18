from typing import List

import numpy as np
import torch


class ProdOut:
    """
    You can use this class as a starting point to implement your "real use case"
    """

    def out(self, predictions: torch.Tensor, names: List[str], frames: List[np.ndarray]):
        """
        This is the place where you implement your out-logic.
        notice that you'll receive a list of predictions and filenames. The list list size equals your batch size.

        You will probably want to find the most probable class first
        or do more complex things to get more accurate class assignments

        Next step will be mapping the predicted class in a format suitable for your use case.
        Eg for image export this would be image coordinates

        Args:
            predictions: network result (list of samples containing probabilities per sample)
            names: filenames for predictions, might not be available depending on input module
            frames: source frames, might not be available depending on input module
        """
        for i in range(0, len(predictions)):
            print(predictions[i])

    def post(self):
        """
        Called after dataset/video/whatever was completely processed. You can do things like cleanup or printing stats here
        """
        print("finished")
