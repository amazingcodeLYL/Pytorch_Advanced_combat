import torch
import torchvision
from torchvision import datasets,transforms
from torch.autograd import Variable
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform=transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((.5,.5,.5),(.5,.5,.5)) #将数据进行标准化转换
])
data_train=datasets.MNIST(root='./data/',transform=transform,train=True,download=True)
data_test=datasets.MNIST(root='./data/',transform=transform,train=False)

train_loader=DataLoader(dataset=data_train,batch_size=64,shuffle=True)
test_loader=DataLoader(dataset=data_test,batch_size=64,shuffle=True)


images,labels=next(iter(train_loader))
img = torchvision.utils.make_grid(images)
print(images)
plt.figure()
img = img.numpy().transpose(1, 2, 0)
# mean = [0.5, 0.5, 0.5]
# std = [0.5, 0.5, 0.5]
# img = img * std + mean
print([labels[i] for i in range(64)])
plt.imshow(img)
plt.show()

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(stride=2,kernel_size=2)
        )
        self.dense=nn.Sequential(
            nn.Linear(14*14*128,1024),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024,10)
        )
    def forward(self,x):
        x=self.conv1(x)
        x=x.view(-1,14*14*128)
        x=self.dense(x)
        return x

model=Model().to(device)
cost=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters())
n_epochs=5

for epoch in range(n_epochs):
    running_loss=0
    running_correct=0
    print("Epoch{}/{}".format(epoch,n_epochs))
    print("*"*10)
    for data in train_loader:
        images,labels=data
        images,labels=Variable(images),Variable(labels)
        images, labels =images.to(device),labels.to(device)
        y_pred=model(images)
        _,pred=torch.max(y_pred.data,1)
        optimizer.zero_grad()
        loss=cost(y_pred,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
        running_correct+=torch.sum(pred==labels.data)
    testing_correct=0
    for data in test_loader:
        images,labels=data
        images, labels = Variable(images), Variable(labels)
        images, labels = images.to(device), labels.to(device)
        y_pred=model(images)
        _, pred = torch.max(y_pred.data, 1)
        testing_correct+=torch.sum(pred==labels.data)
    print("Train Loss:{:4f},Train Accuracy:{:.4f}%,Test Accuracy:{:.4f}".format(running_loss/len(data_train),100*running_correct/len(data_train),100*testing_correct/len(data_test)))



###################
#测试
test_image=DataLoader(dataset=data_test,batch_size=4,shuffle=True)
X_test,y_test=next(iter(test_image))
inputs=Variable(X_test)
inputs=inputs.to(device)
pred=model(inputs).to(device)
_,pred=torch.max(pred,1)
print(pred)
print("Predict label is :",[i for i in pred.data])
print("Real Label is :",[i for i in y_test])

img=torchvision.utils.make_grid(X_test)
img=img.numpy().transpose(1,2,0)
plt.imshow(img)
plt.show()