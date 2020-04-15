import pandas as pd
import numpy as np

# 数据预处理
def dataProcess(df):
    x_list, y_list = [], []
    df = df.replace(['NR'], [0.0])
    array = np.array(df).astype(float)
    # 将数据集拆分为多个数据帧
    for i in range(0, 4320, 18):
        for j in range(24-9):
            mat = array[i:i+18, j:j+9]
            label = array[i+9, j+9] # 第10行是PM2.5
            x_list.append(mat)
            y_list.append(label)
    x = np.array(x_list)
    y = np.array(y_list)

    '''
    # 将每行数据都scale到0到1的范围内，有利于梯度下降，但经尝试发现效果并不好
    for i in range(18):
        if(np.max(x[:, i, :]) != 0):
            x[: , i, :] /= np.max(x[:, i, :])
    '''
    return x, y, array

def train(x_train,y_train,epoch):
    bias=0
    weights=np.ones(9)
    learning_rate=1
    reg_rate=0.001
    bg2_sum=0
    wg2_sum=np.zeros(9)

    for i in range(epoch):
        b_g=0
        w_g=np.zeros(9)
        for j in range(3200):
            b_g+=(y_train[j]-weights.dot(x_train[j,9,:])-bias)*(-1)
            for k in range(9):
                w_g[k]+=(y_train[j]-bias-weights.dot(x_train[j,9,:]))*(-x_train[j,9,k])

        b_g/=3200
        w_g/=3200

        for m in range(9):
            w_g[m]+=reg_rate*weights[m]

        bg2_sum+=b_g**2
        wg2_sum+=w_g**2

        bias-=learning_rate/bg2_sum**0.5*b_g
        weights-=learning_rate/wg2_sum**0.5*w_g

        if i %200==0:
            loss=0
            for j in range(3200):
                loss+=(y_train[j]-weights.dot(x_train[j,9,:])-bias)**2
            print('after {} epochs,the loss is :'.format(i),loss/3200)

    return  weights,bias

def validate(x_val,y_val,weights,bias):
    loss=0
    for i in range(400):
        loss+=(y_val[i]-weights.dot(x_val[i,9,:])-bias)**2
        return loss/400

def main():
    df=pd.read_csv('E:/train.csv',usecols=range(3,27))
    x,y,_=dataProcess(df)
    x_train,y_train=x[0:3200],y[0:3200]
    x_val,y_val=x[3200:3600],y[3200:3600]
    epoch=2000
    w,b=train(x_train,y_train,epoch)
    loss=validate(x_val,y_val,w,b)
    print('The loss on val data is :',loss)

if __name__=='__main__':
    main()