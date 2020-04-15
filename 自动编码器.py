import torch
import torchvision
from torchvision import datasets,transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],std=[0.5])
])
datasets_train=datasets.MNIST(root='./data',train=True,transform=transform,download=True)
datasets_test=datasets.MNIST(root='./data',train=False,transform=transform)

train_load=DataLoader(dataset=datasets_train,batch_size=4,shuffle=True)
test_load=DataLoader(dataset=datasets_test,batch_size=4,shuffle=True)

# imaegs,labels=next(iter(train_load))
# print(imaegs.shape)
# imaegs_example=torchvision.utils.make_grid(imaegs)
# imaegs_example=imaegs_example.numpy().transpose(1,2,0)
# mean=[0.5,0.5,0.5]
# std=[0.5,0.5,0.5]
# imaegs_example=imaegs_example*std+mean
# print(imaegs_example.shape)
# plt.imshow(imaegs_example)
# plt.show()

# x=imaegs_example.shape
# noisy_images=imaegs_example+0.5*np.random.randn(x[0],x[1],x[2])
# noisy_images=np.clip(noisy_images,0.,1.)
# plt.imshow(noisy_images)
# plt.show()

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder=nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU()
        )
        self.decoder=nn.Sequential(
            nn.Linear(32,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,28*28)
        )
    def forward(self,input):
        encoder=self.encoder(input)
        output=self.decoder(encoder)
        return output

# device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu=torch.cuda.is_available()
model=AutoEncoder()
print(model)

# if use_gpu:
#     model=model.cuda()

optimizer=optim.Adam(model.parameters())
loss_f=nn.MSELoss()
epoch_n=10
for epoch in range(epoch_n):
    running_loss=0.0
    print("Epoch {}/{}".format(epoch,epoch_n))
    print("-"*10)
    for data in train_load:
        X,_=data
        X=Variable(X)

        noisy_X_train=X+0.5*torch.randn(X.shape)
        noisy_X_train=torch.clamp(noisy_X_train,0.,1.0)
        noisy_X_train=Variable(noisy_X_train)
        X=X.view(-1,28*28)
        noisy_X_train=noisy_X_train.view(-1,28*28)
        train_pre=model(noisy_X_train)
        loss=loss_f(train_pre,X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print("Loss is：{:.4f}".format(running_loss/len(datasets_train)))

X_test,_=next(iter(test_load))
img1=torchvision.utils.make_grid(X_test)
imaegs_example=img1.numpy().transpose(1,2,0)
mean=[0.5,0.5,0.5]
std=[0.5,0.5,0.5]
img1=img1*std+mean
noisy_X_test=img1+0.5*np.random.randn(*img1.shape)
noisy_X_test=np.clip(noisy_X_test,0.,1.)
plt.figure()
plt.imshow(noisy_X_test)

img2=X_test+0.5*torch.randn(*X_test.shape)
img2=torch.clamp(img2,0.,1.)

img2=Variable(img2.view(-1,28*28))
test_pred=model(img2)

img_test=test_pred.data.view(-1,1,28,28)

img2=torchvision.utils.make_grid(img_test)
img2=img2.numpy().transpose(1,2,0)
img2=img2*std+mean
img2=np.clip(img2,0.,1.)
plt.figure()
plt.imshow(img2)