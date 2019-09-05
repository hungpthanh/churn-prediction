from sklearn import tree
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, input_dim, output_dim, classes_number):
        super(FFN, self).__init__()

        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, classes_number)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.relu(x)
        x = F.log_softmax(x, dim=1)
        return x



