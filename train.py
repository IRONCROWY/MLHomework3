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

from net import AlexNet,SimpleNet,Activation_Net,Batch_Net,FC3, createDenseNet
from dataloader import data_set,mydataset,testdataset

def flip(x,dim):
    dim=x.dim()+dim if dim<0 else dim
    inds=tuple(slice(None,None) if i!=dim else x.new(torch.arange(x.size(i)-1,-1,-1).tolist()).long() for i in range(x.dim()))
    return x[inds] 


#--------------------------------------------------------------------------------------------------------------------------------------

train_set,test_set=mydataset().train_test_split()

BATCH_SIZE=4
NUM_EPOCHS=50
LEARNING_RATE=1e-5

train_loader=data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader=data.DataLoader(test_set, batch_size=1, shuffle=False, drop_last=True)

# cont_list=[]
# for img,index in tqdm(ttest_loader):
#     cont_list=cont_list+[{"name":testing_list[index],"result":0}]
# df=pd.DataFrame(cont_list)
# df.to_csv("testresult.csv",index=False)


model=AlexNet()
# model=createDenseNet(nb_classes=2,img_dim=(40,40,40,1))

if torch.cuda.is_available:
    model=model.cuda()
    device=torch.device('cuda:0')

criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=LEARNING_RATE,betas=(0.9,0.999))
# optimizer=optim.SGD(model.parameters(),lr=LEARNING_RATE)

train_acc=0
epoch=0
epoch_acc=0
print_loss=0

acc1=0
acc2=0
acc3=0

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_acc=0
    for img,label in tqdm(train_loader,ncols=50):
        # img=img.view(img.size(0),-1)
        img=img.to(device)
        label=label.to(device)
        #1
        out=model(img)

        loss1=criterion(out,label)
        # loss1=torch.mean(torch.clamp(1-out.t()*label,min=0))
        print_loss1=loss1.data.item()

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        _,pred=torch.max(out,1)
        num_correct=(pred==label).sum()
        train_acc+=num_correct.item()
        epoch_acc+=num_correct.item()
        #2
        
        out=model(flip(img,1))

        loss2=criterion(out,label)
        # loss2=torch.mean(torch.clamp(1-out.t()*label,min=0))
        print_loss2=loss2.data.item()

        optimizer.zero_grad()
        loss2.backward()
        optimizer.step()

        _,pred=torch.max(out,1)
        num_correct=(pred==label).sum()
        train_acc+=num_correct.item()
        epoch_acc+=num_correct.item()
        #3
        out=model(flip(img,2))

        loss3=criterion(out,label)
        # loss3=torch.mean(torch.clamp(1-out.t()*label,min=0))
        print_loss3=loss3.data.item()

        optimizer.zero_grad()
        loss3.backward()
        optimizer.step()

        _,pred=torch.max(out,1)
        num_correct=(pred==label).sum()
        train_acc+=num_correct.item()
        epoch_acc+=num_correct.item()
        #4
        out=model(flip(img,3))

        loss4=criterion(out,label)
        # loss4=torch.mean(torch.clamp(1-out.t()*label,min=0))
        print_loss4=loss4.data.item()

        optimizer.zero_grad()
        loss4.backward()
        optimizer.step()

        _,pred=torch.max(out,1)
        num_correct=(pred==label).sum()
        train_acc+=num_correct.item()
        epoch_acc+=num_correct.item()


    epoch_acc=epoch_acc/4
    print_loss=(print_loss1+print_loss2+print_loss3+print_loss4)/4





    epoch_acc=epoch_acc/len(train_set)

    

    model.eval()
    eval_loss=0
    eval_acc=0
    for img,label in tqdm(test_loader,ncols=50):
        img=img.to(device)
        label=label.to(device)
        # img=img.view(img.size(0),-1)
        out=model(img)
        loss=criterion(out,label)
        # loss=torch.mean(torch.clamp(1-out.t()*label,min=0))

        eval_loss+=loss.data.item()*label.size(0)
        _, pred=torch.max(out,1)
        num_correct=(pred==label).sum()
        eval_acc+=num_correct.item()
    

    state={'net':model.state_dict()}
    if ((train_acc/((len(train_set))*(epoch+1))*4)>=acc1 and eval_acc/(len(test_set))>=acc2 and epoch_acc>=acc3):
        acc1=train_acc/((len(train_set))*(epoch+1)*4)
        acc2=eval_acc/(len(test_set))
        acc3=epoch_acc
        tmp_epoch=epoch+1
        torch.save(state,'C3D.tar')
        print("SAVED!")

    print('EPOCH: {}, LOSS: {:.4}, Train_Acc:{:.6f}, Epoch_Acc:{:.6f}, Highest_Acc:{:.6f}, Saved_Epoch:{}'.format(epoch+1, loss.data.item(), train_acc/((len(train_set))*(epoch+1)*4), epoch_acc, acc1, tmp_epoch))
    print('Test Loss:{:.6f}, Acc:{:.6f}\n'.format(eval_loss/(len(test_set)),eval_acc/(len(test_set))))

print('Acc:{:.6f}'.format(train_acc/(len(train_set)*(NUM_EPOCHS))))       


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
