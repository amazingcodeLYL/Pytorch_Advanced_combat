import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
batch_n=64
hidden_layer=100
input_data=1000
output_data=10


models=nn.Sequential(
    nn.Linear(input_data,hidden_layer),
    nn.ReLU(),
    nn.Linear(hidden_layer,output_data)
)

x=Variable(torch.randn(batch_n,input_data),requires_grad=False)
y=Variable(torch.randn(batch_n,output_data),requires_grad=False)

w1=Variable(torch.randn(input_data,hidden_layer),requires_grad=True)
w2=Variable(torch.randn(hidden_layer,output_data),requires_grad=True)

epoch_n=1000
learning_rate=1e-4
loss_fn=nn.MSELoss()
optimzer=optim.Adam(models.parameters(),lr=learning_rate)

for epoch in range(epoch_n):
    y_pred=models(x)

    loss=loss_fn(y_pred,y)
    print("Epoch:{},Loss:{:.4f}".format(epoch,loss.item()))

    optimzer.zero_grad()
    loss.backward()
    optimzer.step()



