import torch
import numpy as np

class DataLoader():
    def __init__(self, dataset_input, dataset_output, batch_size=50):
        """
        Loading data for Deep Learning Model
        :param dataset_input: training data
        :param dataset_output: target data
        :param batch_size: The size of each batch
        """
        self.dataset_input = dataset_input
        self.dataset_output = dataset_output
        self.batch_size = batch_size

    def __iter__(self):
        L = (self.dataset_input.shape[0] // self.batch_size) * self.batch_size
        for st in range(0, L, self.batch_size):
            input = self.dataset_input[st: st + self.batch_size]
            output = self.dataset_output[st: st + self.batch_size]
            yield input, output