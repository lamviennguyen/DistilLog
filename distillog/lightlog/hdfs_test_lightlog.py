from email import header
import pandas as pd
import numpy as np
import time
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

'''
    This step reduces the dimensionality of the 300-dimensional semantic vector into 20-dimensional data. 
    Due to the overwhelming size of the original dataset, a sample of 2000 was selected for presentation.
'''
split = 50
path = "../datasets/HDFS/test.csv"

ppa_result = pd.read_csv("../datasets/HDFS/hdfs_semantic_vector.csv", header = None)
ppa_result = np.array(ppa_result)

# test data

logs_series = pd.read_csv(path, header=None)
logs_series = logs_series.values
total = len(logs_series)
sub = int(total/split)

def mod(l, n):
    """ Truncate or pad a list """
    r = l[-1*n:]
    if len(r) < n:
        r.extend(list([0]) * (n - len(r)))
    return r


def read_test(i):
    if i==split-1:
        label = logs_series[i*sub:, 1]
        logs_data = logs_series[i*sub:, 0]
    else:
        label = logs_series[i*sub:(i+1)*sub, 1]
        logs_data = logs_series[i*sub:(i+1)*sub, 0]
    logs = []

    for logid in range(0,len(logs_data)):
        ori_seq = [
            int(eventid) for eventid in logs_data[logid].split()]
        seq_pattern = mod(ori_seq, 300)
        vec_pattern = []

        for event in seq_pattern:
            if event == 0:
                vec_pattern.append([0]*20)
            else:
                vec_pattern.append(ppa_result[event-1])  
        logs.append(vec_pattern)
    logs = np.array(logs)
    train_x = logs
    train_y = label
    train_x = np.reshape(train_x, (train_x.shape[0], -1, 20))
    train_y = train_y.astype(int)
    return train_x, train_y


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding='same', dilation=dilation_rate)
        self.conv2 = nn.Conv1d(
            out_channels, 1, 3, padding='same', dilation=dilation_rate)
        self.conv3 = nn.Conv1d(in_channels, out_channels,
                               kernel_size, padding='same')
        self.relu = nn.ReLU()

    def forward(self, x):
        r = self.conv1(x)
        r = self.relu(r)
        r = self.conv2(r)
        if x.shape[1] == self.out_channels:
            shortcut = x
        else:
            shortcut = self.conv3(x)
        o = r + shortcut
        o = self.relu(o)
        return o


class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.resblock1 = ResBlock(
            in_channels=20, out_channels=3, kernel_size=3, dilation_rate=1)
        self.resblock2 = ResBlock(
            in_channels=3, out_channels=3, kernel_size=3, dilation_rate=2)
        self.resblock3 = ResBlock(
            in_channels=3, out_channels=3, kernel_size=3, dilation_rate=4)
        self.resblock4 = ResBlock(
            in_channels=3, out_channels=3, kernel_size=3, dilation_rate=8)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(3, 2)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class TCNDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X = torch.moveaxis(self.X, 2, 1)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.length = self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.length


def test_model(test_x, label):
    model = TCN()
    model.to(device)
    model.load_state_dict(torch.load("../datasets/HDFS/model/lightlog.pth",map_location=torch.device('cpu')))
    model.eval()
    test_loader = DataLoader(TCNDataset(test_x, label), batch_size=512)
    y_pred = []
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        y_pred.extend(pred.detach().cpu().numpy())
    return y_pred


# Detect
tp = 0
fp = 0
tn = 0
fn = 0
start_detect = time.perf_counter()
for i in range (0, split):
    test_x, label = read_test(i)
    y_pred = test_model(test_x, label.astype('float'))


# Input result
    y_pred = np.argmax(y_pred, axis=1)

    for j in range(0, len(y_pred)):
        if label[j] == y_pred[j] and label[j] == 1:
            tp = tp+1
        elif label[j] != y_pred[j] and label[j] == 1:
            fp = fp+1
        elif label[j] == y_pred[j] and label[j] == 0:
            tn = tn + 1
        elif label[j] != y_pred[j] and label[j] == 0:
            fn = fn + 1


end_detect = time.perf_counter()
print('The detection time is', end_detect-start_detect)

print('TP,FP,TN,FN are: ', [tp, fp, tn, fn])
print('Precision, Recall, F1-measure are:', tp/(tp+fp), tp /
      (tp+fn), 2*(tp/(tp+fp)*(tp/(tp+fn))/((tp/(tp+fp))+(tp/(tp+fn)))))

