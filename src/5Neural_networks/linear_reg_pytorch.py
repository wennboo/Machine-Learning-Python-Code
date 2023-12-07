
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
# import os

### 创建线性回归网络模型

class LRNet(nn.Module):
    def __init__(self,n_feature, n_hidden, n_output):
        super(LRNet,self).__init__()
        self.hidden=T.nn.Linear(n_feature, n_hidden) #隐藏层
        self.output=T.nn.Linear(n_hidden, n_output) #输出层
        self.optimizer = optim.Adam(self.parameters(), lr=0.2)

    # 前向传播
    def forward(self, x): 
        x=F.relu(self.hidden(x)) #relu激活函数
        x=self.output(x)
        return x


if __name__ == '__main__':

    ITERATION = 100
    ###创建数据集
    uLen = 100
    ##T.linspace(-1,1,uLen),-1到1之间均匀取点
    ##T.unsqueeze实现维数扩充，指定位置加上维数为1的维度，把一维换成二维，变成批量，torch要求
    x=T.unsqueeze(T.linspace(-1,1,uLen),dim=1)
    print('linspace',T.linspace(-1,1,uLen))
    print('x',x)
    print('x.size',x.size())
    y=x.pow(2)+0.2*T.rand(x.size())

    ####创建网络
    lrnet = LRNet(n_feature=1, n_hidden=10, n_output=1)
    #### 确定损失函数
    loss_func= nn.MSELoss()


    plt.ion() #画图转为显示模式

    for t in range(ITERATION):

        lrnet.optimizer.zero_grad() #梯度清零
        prediction=lrnet(x) #前向传播
        loss=loss_func(prediction,y) #求损失函数
        loss.backward() #反向传播
        lrnet.optimizer.step() #更新参数

        if t%5==0:
            plt.cla()
            plt.scatter(x.data.numpy(),y.data.numpy())
            plt.plot(x.data.numpy(),prediction.data.numpy(),'r-',lw=5)
            plt.text(0.5,0,'Loss=%.4f'%loss.data.numpy(),fontdict={'size':20,'color':'red'})
            plt.pause(0.1)


    plt.ioff() #不加会一闪而过
    plt.show() #显示所有图片