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

ttest_set=testdataset()

ttest_loader=data.DataLoader(ttest_set,batch_size=1,shuffle=False,drop_last=True)

model=AlexNet()

checkpoint=torch.load('C3D.tar')
model.load_state_dict(checkpoint['net'])

if torch.cuda.is_available:
    model=model.cuda()
    device=torch.device('cuda:0')

model.eval()
cont_list=[]
for img,name in tqdm(ttest_loader,ncols=80):
    img=img.to(device)
    # img=img.view(img.size(0),-1)
    out=model(img)
    prob=out.data.cpu().numpy()[0][1]

    cont_list=cont_list+[{"ID":name[0],"Predicted":prob}]
df=pd.DataFrame(cont_list)
df.to_csv("testresult.csv",index=False)