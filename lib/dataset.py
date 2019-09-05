import torch
import numpy as np

class DataLoader():
    def __init__(self, dataset_input, dataset_output, batch_size=50):
        """
        A class for creating session-parallel mini-batches.
        Args:
             dataset (SessionDataset): the session dataset to generate the batches from
             batch_size (int): size of the batch
        """
        self.dataset_input = dataset_input
        self.dataset_output = dataset_output
        self.batch_size = batch_size

    def __iter__(self):
        """ Returns the iterator for producing session-parallel training mini-batches.
        Yields:
            input (B,): torch.FloatTensor. Item indices that will be encoded as one-hot vectors later.
            target (B,): a Variable that stores the target item indices
            masks: Numpy array indicating the positions of the sessions to be terminated
        """

        # initializations
        L = (self.dataset_input.shape[0] // self.batch_size) * self.batch_size
        for st in range(0, L, self.batch_size):
            input = self.dataset_input[st: st + self.batch_size]
            output = self.dataset_output[st: st + self.batch_size]
            # output = np.expand_dims(output, axis=1)
            yield input, output