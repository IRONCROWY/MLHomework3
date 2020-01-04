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

from dataloader import data_transforms

from PIL import Image

train_set,test_set=mydataset().train_test_split(p=0)

test_loader=data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True, num_workers=0)

criterion=nn.CrossEntropyLoss()

model=AlexNet()
if torch.cuda.is_available:
    model=model.cuda()
    device=torch.device('cuda:0')

checkpoint=torch.load('C3D.tar')
model.load_state_dict(checkpoint['net'])

model.eval()
eval_loss=0
eval_acc=0
for img,label in tqdm(test_loader,ncols=50):
    img=img.to(device)
    label=label.to(device)
    # img=img.view(img.size(0),-1)
    out=model(img)
    loss=criterion(out,label)

    eval_loss+=loss.data.item()*label.size(0)
    _, pred=torch.max(out,1)
    num_correct=(pred==label).sum()
    eval_acc+=num_correct.item()
print('Test Loss:{:.6f}, Acc:{:.6f}'.format(eval_loss/(len(test_set)),eval_acc/(len(test_set))))