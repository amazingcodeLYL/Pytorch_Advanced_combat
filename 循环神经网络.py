'''
author：lyl
date：4/15 2020
RNN识别手写数字
Epoch: 10
Loss is: 0.0021,Train accuracy is: 96.0000%,Test Accuracy: 95.0000%
'''
import torch
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])

dataset_train=datasets.MNIST(root='./data',train=True,transform=transform,download=True)
dataset_test=datasets.MNIST(root='./data',transform=transform,train=False)

train_load=DataLoader(dataset=dataset_train,batch_size=64,shuffle=True)
test_load=DataLoader(dataset=dataset_test,batch_size=64,shuffle=True)

# images,label=next(iter(train_load))
# images_example=torchvision.utils.make_grid(images)
# images_example=images_example.numpy().transpose(1,2,0)
# mean=[0.5,0.5,0.5]
# std=[0.5,0.5,0.5]
# images_example=images_example*std+mean
# plt.imshow(images_example)
# plt.show()

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.RNN(input_size=28,hidden_size=128,num_layers=1,batch_first=True)
        #num_layers 指定循环层堆叠的数量，默认为1  RNN中输入层和输出层数据默认维度是(seq,batch,feature)
        # seq序列的长度batch数据批次的数量 feature输入或输出的特征数
        #将batch_first设置为True 输入层和输出层数据维度重新对应为（batch,seq,feature）
        self.output=nn.Linear(128,10)

    def forward(self,input):
        output,_=self.rnn(input,None) #对H0初始化采用0初始化，传入的参数为None
        output=self.output(output[:,-1,:])#需要处理的问题是分类问题，需要提取最后一个序列的输出结果作为RNN的输出
        return output

model=RNN().to(device)
optimizer=optim.Adam(model.parameters())
loss_f=nn.CrossEntropyLoss()
epoch_n=10
for epoch in range(epoch_n):
    running_loss=0
    running_correct=0
    testing_correct=0
    print("Epoch {}/{}".format(epoch,epoch_n))
    print("-"*10)
    for _,data in enumerate(train_load,1):
        X,y=data
        X,y=X.to(device),y.to(device)
        X=X.view(-1,28,28)
        y_pred=model(X)
        loss=loss_f(y_pred,y)
        _,pred=torch.max(y_pred,1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss+=loss.item()
        running_correct+=torch.sum(pred==y.data)


    for data in test_load:
        X,y=data
        X, y = X.to(device), y.to(device)
        X=X.view(-1,28,28)#输入数据维度(batch,seq,feature)
        outputs=model(X)
        _,pred=torch.max(outputs,1)
        testing_correct+=torch.sum(pred==y.data)

print("Loss is: {:.4f},Train accuracy is: {:.4f}%,Test Accuracy: {:.4f}%".format(running_loss/len(dataset_train),running_correct*100/len(dataset_train),testing_correct*100/len(dataset_test)))
