import torch
import torchvision as tv
from torchvision import transforms
import os
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch import optim
import torch.utils.data as data
import time
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import csv

from net import AlexNet,SimpleNet,Activation_Net,Batch_Net,FC3
from dataloader import data_set,mydataset,testdataset

# matrix=np.array([x for x in range(27)])
# matrix=matrix.reshape(3,3,3)

# print(matrix)
# print(matrix.transpose(2,0,1))

criterion=nn.CrossEntropyLoss()
target=torch.Tensor([1])
input=torch.Tensor([[0.5123,0.5887]])
output=criterion(input,target)
print(output)
