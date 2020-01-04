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

Training_Directory="data/train_val"
Training_CSV="data/train_val.csv"
Testing_Directory="data/test"

data_transforms=transforms.Compose([transforms.ToTensor()])

class data_set(torch.utils.data.Dataset):
    def __init__(self,index):
        self.index=index

        self.training_list=np.array(os.listdir(Training_Directory))
        self.sort()
        self.training_list=self.training_list[index]

        self.label_list=pd.read_csv(Training_CSV).values
        self.label=self.label_list[:,1][self.index]
    
    def sort(self):
        t_list=self.training_list
        sorted_list=sorted(t_list,key=lambda x:(int)(os.path.splitext(x)[0].strip('candidate')))
        self.training_list=np.array(sorted_list)
    
    def __getitem__(self,index):
        data=np.load(os.path.join(Training_Directory,self.training_list[index]))
        voxel=data_transforms(data['voxel'].astype(np.float32))/255
        seg=data_transforms(data['seg'].astype(np.float32))
        data=(voxel*seg)[20:80,20:80,20:80]
        

        # data=np.zeros(shape=(4,40,40,40))
        # data[0]=temp
        # data[1]=np.flip(temp,1)
        # data[2]=np.flip(temp,2)
        # data[3]=np.flip(data[1],2)
        # print(data.shape)
        # data=torch.from_numpy(data)
        # data=data.type(torch.FloatTensor)
        # data=data.cuda()

        data=data.unsqueeze(0)

        label=self.label[index]

        return data,label

    def __len__(self):
        return len(self.training_list)
    


class mydataset():
    def __init__(self):

        self.training_list=np.array(os.listdir(Training_Directory))
        self.sort()
    
    def sort(self):
        t_list=self.training_list
        sorted_list=sorted(t_list,key=lambda x:(int)(os.path.splitext(x)[0].strip('candidate')))
        self.training_list=np.array(sorted_list)
    
    def train_test_split(self,p=0.8):

        length=len(self.training_list)
        index_list=np.array(range(length))
        np.random.shuffle(index_list)

        self.train_index=index_list[:(int)(length*p)]
        self.test_index=index_list[(int)(length*p):]

        self.train_set=data_set(self.train_index)
        self.test_set=data_set(self.test_index)

        return self.train_set,self.test_set
    
    # def __getitem__(self,index):
    #     data=np.load(os.path.join(Training_Directory,self.training_list[index]))
    #     voxel=data_transforms(data['voxel'].astype(np.float32))/255
    #     seg=data_transforms(data['seg'].astype(np.float32))
    #     data=(voxel*seg)[30:70,30:70,30:70]

    #     # temp=tmp[30:70,30:70,30:70]
    #     # data=np.zeros(shape=(2,40,40,40))
    #     # data[0]=data[1]=temp
    #     # data=torch.from_numpy(data)
    #     # data=data.type(torch.FloatTensor)
    #     # data=data.cuda()

    #     data=data.unsqueeze(0)

    #     return seg

    # def __len__(self):
    #     return len(self.training_list)



class testdataset(torch.utils.data.Dataset):
    def __init__(self):
        self.testing_list=np.array(os.listdir(Testing_Directory))
        self.sort()
    
    def sort(self):
        t_list=self.testing_list
        sorted_list=sorted(t_list,key=lambda x:(int)(os.path.splitext(x)[0].strip('candidate')))
        self.testing_list=np.array(sorted_list)

    def __getitem__(self,index):
        data=np.load(os.path.join(Testing_Directory,self.testing_list[index]))
        voxel=data_transforms(data['voxel'].astype(np.float32))/255
        seg=data_transforms(data['seg'].astype(np.float32))
        data=(voxel*seg)[20:80,20:80,20:80]

        # data=np.zeros(shape=(4,40,40,40))
        # data[0]=temp
        # data[1]=np.flip(temp,1)
        # data[2]=np.flip(temp,2)
        # data[3]=np.flip(data[1],2)
        # data=torch.from_numpy(data)
        # data=data.type(torch.FloatTensor)
        # data=data.cuda()

        name=os.path.basename(self.testing_list[index])        
        name=os.path.splitext(name)[0] 

        data=data.unsqueeze(0)

        return data,name

    def __len__(self):
        return len(self.testing_list)