import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


#### y=xW+b;
#### loss = sum((y-y')^2)/2;y'是标签
#### 令 grad = d(loss)/dy = (y-y') 表示当前网络的上一层梯度；是个
#### d(loss)/dw = d(loss)/dy * dy/dw = x^T * (y-y')
#### d(loss)/db = (y-y')
#### d(loss)/dx = d(loss)/dy * dy/dx =  (y-y') *w



class LinearLayer:
    def __init__(self,input_D,output_D):
        self._W =np.random.normal(0,0.1,(input_D,output_D))#初始化不能为全g
        self._b=np.random.normal(0,0.1,(1,output_D))
        self._grad_W=np.zeros((input_D,output_D))
        self._grad_b=np.zeros((1,output_D))

    def forward(self,X):
        return np.matmul(X,self._W)+self._b

    def backward(self,X,grad): #本层的输入
        self._grad_W=np.matmul(X.T, grad)
        self._grad_b=np.matmul(grad.T, np.ones(X.shape[0]))
        return np.matmul(grad, self._W.T)

    def update(self,learn_rate): #梯度下降法
        self._W=self._W-self._grad_W*learn_rate
        self._b=self._b-self._grad_b*learn_rate



class Sigmoid:
    def __init__(self):
        pass
    def forward(self,X):
        return 1/(1+np.exp(-X))
    def backward(self,X,grad):
        tem=1/(1+np.exp(-X))
        tem= np.multiply(tem, 1-tem)
        return np.multiply(tem, grad)

class Relu:
    def __init__(self):
        pass
    def forward(self,X):
        return np.where(X>0,X,0)

    def backward(self,X,grad):
        return np.where(X>0,1,0)*grad

class LeakyRelu:
    def __init__(self):
        pass
    def forward(self,X):
        return np.where(X>0,X,0.001*X)

    def backward(self,X,grad):
        return np.where(X>0,1,0.001)*grad

class Linear:
    def __init__(self):
        pass
    def forward(self,X):
        return X
        
    def backward(self,X,grad):
        return grad


if __name__ == '__main__':

    #训练参数设置
    STEP = 1000
    P_STEP = 10
    learn_rate = 0.001
    err = 0.01
    uLen = 1000 #样本数
    #样本产生
    x=np.linspace(-1,1,uLen)
    train_X = x.reshape(x.shape[0],-1)#转成批量样本
    #生成标签
    y_=np.power(train_X,2)+0.2*np.random.uniform(0,1,size=train_X.shape)


    #初始化网络，确定输入输出维数，确定激活函数
    input_D = train_X.shape[1]
    hidden_D = 10
    output_D = y_.shape[1]

    linear1=LinearLayer(input_D,hidden_D)
    # n_act=Sigmoid()
    n_act = Relu()
    # n_act = LeakyRelu()
    linear2=LinearLayer(hidden_D,y_.shape[1])
    ###画图设置
    plt.ion() #画图转为显示模式
    for i in range(STEP):
        #前向传播Forward,获取网络输出
        x = train_X
        alpha = linear1.forward(x)
        b = n_act.forward(alpha)
        beta = linear2.forward(b)
        y=beta
        # print('y = ',y)
        # y = n_act.forward(beta)
        #计算损失函数
        loss= mean_squared_error(y,y_)#MsE损失函数
        # print('loss = %.4f'%(loss))
        #反向传播，获取梯度(完全根据链式法则来)
        grad=y -y_ #第一步
        # grad=n_act.backward(beta,grad)  #输出激活函数
        # print('hello',grad)
        grad=linear2.backward(b,grad) #输出层
        grad=n_act.backward(alpha,grad) #隐层层激活函数
        grad=linear1.backward(x,grad)
        #更新网络中线性层的参数
        linear1.update(learn_rate)
        print()
        linear2.update(learn_rate)


        #判断学习是否完成
        if loss<err:
            print('loss = %.4f'%(loss))
            print("训练完成！第%d次迭代"%(i))
            break



        if i%P_STEP==0:
            plt.cla()
            plt.scatter(x,y_)
            plt.plot(x,y,'r-',lw=5)
            plt.text(0.5,0,'Loss=%.4f'%loss,fontdict={'size':20,'color':'red'})
            plt.pause(0.1)

        if i%P_STEP==0:
            print('loss = %.4f'%(loss))
    
    plt.ioff() #不加会一闪而过
    plt.show() #显示所有图片








