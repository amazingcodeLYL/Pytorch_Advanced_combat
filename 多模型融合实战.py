'''
author:lyl
date:4/13 2020
多模型融合 Epoch 4 Blending_Model ACC:98.0000
'''

import torch
from torchvision import transforms,datasets
import os
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable

data_dir="..\Include\data\DogsVSCats"
data_transform={x:transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])
    for x in ["train",'valid']}

image_dataset={x:datasets.ImageFolder(root=os.path.join(data_dir,x),transform=data_transform[x])
               for x in ["train",'valid']}
dataloader={x:DataLoader(dataset=image_dataset[x],batch_size=16,shuffle=True) for x in ["train","valid"]}
# X_example,y_example=next(iter(dataloader["train"]))
# example_classes=image_datasets["train"].classes
# index_classes=image_datasets["train"].class_to_idx
# print(example_classes,index_classes)



model_1=models.vgg16(pretrained=True)
model_2=models.resnet50(pretrained=True)
for param in model_1.parameters():
    param.required_grad=False

model_1.fc=nn.Linear(4096,2)

for param in model_2.parameters():
    param.required_grad=False
model_2.fc=nn.Linear(2048,2)


use_gpu=torch.cuda.is_available()
if use_gpu:
    model_1=model_1.cuda()
    model_2=model_2.cuda()


loss_f_1=nn.CrossEntropyLoss()
loss_f_2=nn.CrossEntropyLoss()
optimizer_1=optim.Adam(model_1.fc.parameters(),lr=1e-5)
optimizer_2=optim.Adam(model_2.fc.parameters(),lr=1e-5)
weight_1=0.6
weight_2=0.4

epoch_n=5
time_open=time.time()

for epoch in range(epoch_n):
    print("Epoch {}/{}".format(epoch,epoch_n-1))
    print("-"*10)

    for phase in ["train", "valid"]:
        if phase == "train":
            print("Training.....")
            model_1.train(True)
            model_2.train(True)
        else:
            print("Validing.....")
            model_1.train(False)
            model_2.train(False)

        running_loss_1 = 0.0
        running_corrects_1 = 0
        running_loss_2 = 0
        running_corrects_2 = 0
        blending_running_corrects = 0

        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            if use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(x), Variable(y)

            y_pred_1 = model_1(X)
            y_pred_2 = model_2(X)
            blending_running_corrects = y_pred_1 * weight_1 + y_pred_2 * weight_2

            _, pred_1 = torch.max(y_pred_1)
            _, pred_2 = torch.max(y_pred_2)
            _, blending_pred = y_pred_1 * weight_1 + y_pred_2 * weight_2

            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            loss_1 = loss_f_1(y_pred_1, y)
            loss_2 = loss_f_2(y_pred_2, y)
            if phase == "train":
                loss_1.backward()
                loss_2.backward()
                optimizer_1.step()
                optimizer_2.step()
            running_loss_1 += loss_1.item()
            running_corrects_1 += torch.sum(pred_1 == y.data)
            running_loss_2 += loss_2.item()
            running_corrects_2 += torch.sum(pred_2 == y.data)
            blending_running_corrects += torch.sum(blending_pred == y.data)
            if batch % 500 == 0 and phase == "train":
                print("Batch {},Model1 Train Loss:{:.4f},Model1 Train ACC:{:.4f},Model2 Train Loss:{:.4f},"
                      "Model2 Train ACC:{:.4f},Blending_Model ACC:{:.4f}".format(batch, running_loss_1 / batch,
                                                                                 running_corrects_1 * 100 / (
                                                                                             16 * batch),
                                                                                 running_loss_2 / batch,
                                                                                 100 * running_corrects_2 / (
                                                                                             16 * batch),
                                                                                 blending_running_corrects * 100 / (
                                                                                             16 * batch)))
        epoch_loss_1=running_loss_1*16/len(image_datasets[phase])
        epoch_acc_1=100*running_corrects_1/len(image_datasets[phase])
        epoch_loss_2 = running_loss_2 * 16 / len(image_datasets[phase])
        epoch_acc_2 = 100 * running_corrects_2 / len(image_datasets[phase])
        epoch_blending_acc=100*blending_running_corrects/len(image_datasets[phase])

        print("Epoch ,Model1 Loss:{:.4f},Model1 Acc::{:.4f}%,Model2 Loss:{:.4f},Model2 Acc::{:.4f}%,Blending_Model ACC:{:.4f}".format(epoch_loss_1,epoch_acc_1,epoch_loss_2,epoch_acc_2,epoch_blending_acc))
        time_end=time.time()-time_open
        print(time_end)



