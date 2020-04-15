import numpy as np
import pandas as pd

def train(x_train,y_train,epoch):
    num=x_train.shape[0]
    dim=x_train.shape[1]
    learning_rate=1.0
    weights=np.ones(dim)
    reg_rate=0.001
    bias=0
    bg2_sum=0
    wg2_sum=np.zeros(dim)
    for i in range(epoch):
        b_g = 0
        w_g = np.zeros(dim)
        for j in range(num):
            y_pre=weights.dot(x_train[j,:])+bias
            sig=1/(1+np.exp(-y_pre))
            b_g+=(-1)*(y_train[j]-sig)
            for k in range(dim):
                w_g[k]+=(-1)*x_train[j,k]*(y_train[j]-sig)+2*reg_rate*weights[k]
        b_g/=num
        w_g/=num


        #adagrad
        bg2_sum+=b_g**2
        wg2_sum+=w_g**2

        # bias-= learning_rate / bg2_sum ** 0.5 * b_g
        # weights-= learning_rate / wg2_sum ** 0.5 * w_g
        bias -= learning_rate / bg2_sum ** 0.5 * b_g
        weights -= learning_rate / wg2_sum ** 0.5 * w_g

        if i % 3==0:
            acc=0.0
            result=np.zeros(num)
            for j in range(num):
                y_pre=weights.dot(x_train[j,:])+bias
                sig=1/(1+np.exp(-y_pre))
                if sig>=0.5:
                    result[j]=1
                else:
                    result[j]=0

                if result[j]==y_train[j]:
                    acc+=1.0
            print('after {} epochs ,the acc on train data is :'.format(i),acc/num)
    return weights,bias

def validate(x_val,y_val,weights,bias):
    num=500
    acc=0
    result=np.zeros(num)
    for j in range(num):
        y_pre=weights.dot(x_val[j,:])+bias
        sig=1/(1+np.exp(-y_pre))
        if sig>=0.5:
            result[j]=1
        else:
            result[j]=0

        if result[j]==y_val[j]:
            acc+=1.0

    return acc/num




def main():
    df=pd.read_csv('E:/spam_train.csv')
    df.fillna(0)
    array=np.array(df)
    x=array[:,1:-1]
    x[:,-1]/=np.mean(x[:,-1])
    x[:,-2]/=np.mean(x[:,-2])
    y=array[:,-1]
    x_train,x_val=x[0:3500,:],x[3500:4000,:]
    y_train,y_val=y[0:3500],y[3500:4000]
    epoch=30
    w,b=train(x_train,y_train,epoch)
    acc=validate(x_val,y_val,w,b)
    print('The acc on validate data is :',acc)

if __name__=='__main__':
    main()