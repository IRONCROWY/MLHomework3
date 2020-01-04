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

class SimpleNet(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(SimpleNet,self).__init__()
        self.layer1=nn.Linear(in_dim,n_hidden_1)
        self.layer2=nn.Linear(n_hidden_1,n_hidden_2)
        self.layer3=nn.Linear(n_hidden_2,out_dim)

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

class Activation_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Activation_Net,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.ReLU(True))
        self.layer3=nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x

class Batch_Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Batch_Net,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1),nn.ReLU(True))
        self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2),nn.ReLU(True))
        self.layer3=nn.Sequential(nn.Linear(n_hidden_2,out_dim))

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        return x


# 3D-CNN
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        
        self.conv1 = nn.Conv3d(1, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool3a = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3b = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3b = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv3c = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3c = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv4c = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        # self.conv5a = nn.Conv3d(512, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv5b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        
        # self.fc6 = nn.Linear(5120, 512)
        # self.fc7 = nn.Linear(512, 32)
        # self.fc8 = nn.Linear(32, 2)

        self.fc6 = nn.Linear(16128, 768)
        self.fc7 = nn.Linear(768, 48)
        self.fc8 = nn.Linear(48, 2)
        self.fc=nn.Linear(5120,2)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax=nn.Softmax()

        self.dropout1 = nn.Dropout(p=0.7)
        self.dropout2 = nn.Dropout(p=0.6)
        self.dropout3 = nn.Dropout(p=0.5)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        # x = self.pool3a(x)
        x = self.relu(self.conv3b(x))
        x = self.pool3b(x)
        x = self.relu(self.conv3c(x))
        x = self.pool3c(x)
        

        # x = self.relu(self.conv4a(x))
        # x = self.relu(self.conv4b(x))
        # x = self.relu(self.conv4c(x))
        # x = self.pool4(x)

        # x = self.relu(self.conv5a(x))
        # x = self.relu(self.conv5b(x))
        # x = self.pool5(x)
        # print(x.shape)
        # x = x.view(-1, 5120)
        x = x.view(-1, 16128)
        x = self.relu(self.fc6(x))
        x = self.dropout(x)
        x = self.relu(self.fc7(x))
        x = self.dropout(x)

        logits = self.fc8(x)
        probs=self.softmax(logits)

        return probs

class FC3(nn.Module):
    def __init__(self):
        super(FC3,self).__init__()
        self.fc6 = nn.Linear(40*40*40, 40*40*2)
        self.fc7 = nn.Linear(40*40*2,40*4)
        self.fc8 = nn.Linear(40*4, 8)
        self.fc9 = nn.Linear(8, 2)

        self.dropout1 = nn.Dropout(p=0.7)
        self.dropout2 = nn.Dropout(p=0.6)
        self.dropout3 = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax=nn.Softmax()

    def forward(self,x):
        x = x.view(-1, 40*40*40)
        x = self.relu(self.fc6(x))
        x = self.dropout1(x)
        x = self.relu(self.fc7(x))
        x = self.dropout2(x)
        x = self.relu(self.fc8(x))
        x = self.dropout3(x)

        logits = self.fc9(x)
        probs=self.softmax(logits)
        

        return probs
##_____________________________________________________________________
##DenseNet
from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution3D,Convolution2D
from keras.layers.pooling import AveragePooling3D
from keras.layers.pooling import GlobalAveragePooling3D
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

def conv_block(input, nb_filter, dropout_rate = None, weight_decay = 1E-4):
    x = Activation('relu')(input)
    x = Convolution3D(nb_filter, (3, 3 ,3), kernel_initializer = "he_uniform", padding = "same", use_bias = False,
                      kernel_regularizer = l2(weight_decay))(x)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)

    return x

def transition_block(input, nb_filter, dropout_rate = None, weight_decay = 1E-4):
    concat_axis = 1 if K.image_data_format()=='channels_first' else -1

    x = Convolution3D(nb_filter, (1, 1, 1), kernel_initializer="he_uniform", padding="same", use_bias=False,
                      kernel_regularizer=l2(weight_decay))(input)
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
    x = AveragePooling3D((2, 2,2), strides=(2, 2,2))(x)

    x = BatchNormalization(axis=concat_axis, gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)

    return x

def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    concat_axis = 1 if K.image_data_format()=='channels_first' else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = conv_block(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = Concatenate(axis=concat_axis)(feature_list)
        nb_filter += growth_rate

    return x, nb_filter

def createDenseNet(nb_classes, img_dim, depth = 40, nb_dense_block = 3, growth_rate = 12, nb_filter = 16, dropout_rate=None,
                     weight_decay=1E-4, verbose=True):
    model_input = Input(shape=img_dim)

    concat_axis = 1 if K.image_data_format() == "channels_first" else -1

    assert (depth - 4) % 3 == 0, "Depth must be 3 N + 4"

    # layers in each dense block
    nb_layers = int((depth - 4) / 3)

    # Initial convolution
    x = Convolution3D(nb_filter, (3, 3 ,3), kernel_initializer = "he_uniform", padding = "same", name = "initial_conv3D", use_bias = False,
                      kernel_regularizer = l2(weight_decay))(model_input)

    x = BatchNormalization(axis = concat_axis, gamma_regularizer = l2(weight_decay),
                            beta_regularizer = l2(weight_decay))(x)

    # Add dense blocks
    for block_idx in range(nb_dense_block - 1):
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate = dropout_rate,
                                   weight_decay = weight_decay)
        # add transition_block
        x = transition_block(x, nb_filter, dropout_rate = dropout_rate, weight_decay = weight_decay)

    # The last dense_block does not have a transition_block
    x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate = dropout_rate,
                               weight_decay = weight_decay)

    x = Activation('relu')(x)
    x = GlobalAveragePooling3D()(x)
    x = Dense(nb_classes, activation='softmax', kernel_regularizer=l2(weight_decay), bias_regularizer=l2(weight_decay))(x)

    densenet = Model(inputs=model_input, outputs=x)

    if verbose: 
        print("DenseNet-%d-%d created." % (depth, growth_rate))

    return densenet