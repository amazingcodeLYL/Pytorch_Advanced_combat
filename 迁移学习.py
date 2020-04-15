'''
author:liuyalei
dataset：Kaggle Dogs vs.Cats
Date:2020.4.9
迁移学习
'''
import torch
import torch.nn as nn
import torchvision
import torch.optim as optim
from torchvision import datasets,transforms
import os
import  matplotlib.pyplot as plt
import time
import torchvision.models as models
from torch.utils.data import DataLoader
data_dir="..\Include\data\DogsVSCats"
data_transform={x:transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])
    for x in ["train",'valid']}

image_dataset={x:datasets.ImageFolder(root=os.path.join(data_dir,x),transform=data_transform[x])
               for x in ["train",'valid']}

index_classes=image_dataset["train"].class_to_idx
# print(index_classes)
example_classes=image_dataset["train"].classes
# print(example_classes)



dataloader={x:DataLoader(dataset=image_dataset[x],batch_size=16,shuffle=True) for x in ["train","valid"]}


# X_example,y_example=next(iter(dataloader["train"]))
# print(len(X_example))
# img=torchvision.utils.make_grid(X_example)
# img=img.numpy().transpose([1,2,0])
# print([example_classes[i] for i in y_example])
# plt.imshow(img)
# plt.show()

# class Models(nn.Module):
#     def __init__(self):
#         super(Models, self).__init__()
#         self.Conv=nn.Sequential(
#             nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#
#             nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2,stride=2),
#
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
#
#         self.classes=nn.Sequential(
#             nn.Linear(4*4*512,1024),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(1024,1024),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(1024,2)
#         )
#
#     def forward(self,x):
#         x=self.Conv(x)
#         x=x.view(-1,4*4*512)
#         x=self.classes(x)
#         return x


device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model=models.vgg16(pretrained=True)
# resnet.load_state_dict(torch.load())
for param in model.parameters():
    param.requires_grad=False
model.classifier = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 2)
 )
model=model.to(device)
# print(model)

total_param=sum(p.numel() for p in model.parameters())
print("模型参数总数:{}".format(total_param))
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print('可训练参数总数:{}'.format(total_trainable_params))



criterion=nn.CrossEntropyLoss()
optimzer=optim.Adam(model.classifier.parameters(),lr=1e-5)
epoch_n=5
time_open=time.time()


train_loss=[]
for epoch in range(epoch_n):
    print("Epoch{}/{}".format(epoch,epoch_n-1))
    print("-"*10)

    for p in ["train","valid"]:
        if p=="train":
            print("Training......")
            model.train(True)
        else:
            print("Valid.......")
            model.train(False)

        running_loss=0
        running_correct=0

        for batch,data in enumerate(dataloader[p],1):
            # print(batch)
            images, labels =data
            images,labels=images.to(device),labels.to(device)
            y_pred=model(images)
            _,pred=torch.max(y_pred,1)
            optimzer.zero_grad()
            loss=criterion(y_pred,labels)
            if p=="train":
                loss.backward()
                optimzer.step()
            running_loss+=loss.item()
            running_correct+=torch.sum(pred==labels)
            if batch%500==0 and p=="train":
                print("Batch:{},Train loss:{:.4f},Train Acc:{:.4f}%".format(batch,running_loss/batch,running_correct*100/(16*batch)))
        epoch_loss=running_loss*16/len(image_dataset[p])
        epoch_acc=100*running_correct/len(image_dataset[p])
        print("-"*10)
        print("{} Loss:{:.4f} Acc:{:.4f}%".format(p,epoch_loss,epoch_acc))
        if p == "train":
            train_loss.append(epoch_loss)


plt.plot(train_loss,labels="train")
plt.legend(frameon=False)
plt,show()
time_end=time.time()-time_open
print(time_end)



