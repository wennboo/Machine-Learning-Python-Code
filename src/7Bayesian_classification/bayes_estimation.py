import numpy as np
import pandas as pd
from sklearn.utils.multiclass import type_of_target
from collections import namedtuple


def train_nb(X, y):
    m,n= X.shape #行列
    y_unique_value=y.unique()
    #print(y_unique_value)
    y_nvalue = len(y_unique_value)
    p1 = (len(y[y == '是']) + 1) / (m + y_nvalue) 

    p1_list = []  # 用于保存正例下各属性各取值的条件概率
    p0_list = []

    X1 = X[y == '是'] #好瓜样本提取出来
    X0 = X[y == '否'] #坏瓜样本提取出来

    #print(y == '是')
    m1, _ = X1.shape
    m0, _ = X0.shape
    # print(X1)
    # print(X0)
    # print(p1)

    for i in range(n):
        xi = X.iloc[:, i]#提取属性i的所有行，由Framedata变成Series
        p_xi = namedtuple(X.columns[i], ['is_continuous', 'conditional_pro'])  # 提取属性i建立一个对象,名字为属性i，包含两个属性[]
        #p_xi = namedtuple(xi.name, ['is_continuous', 'conditional_pro'])  # 单列提取标签，提取属性i建立一个对象，对象包含两个属性[]
        is_continuous = type_of_target(xi) == 'continuous'#判定xi的类型

        #print(xi)
        xi1 = X1.iloc[:, i]#提取正样本属性i的所有行，由Framedata变成Series
        xi0 = X0.iloc[:, i]#提取非正样本属性i的所有行，由Framedata变成Series
        if is_continuous:  # 连续值时，conditional_pro 储存的就是 [mean, var] 即均值和方差
            xi1_mean = np.mean(xi1)
            xi1_var = np.var(xi1)
            xi0_mean = np.mean(xi0)
            xi0_var = np.var(xi0)

            p1_list.append(p_xi(is_continuous, [xi1_mean, xi1_var]))
            p0_list.append(p_xi(is_continuous, [xi0_mean, xi0_var]))
        else:  # 离散值时直接计算各类别的条件概率
            unique_value = xi.unique()  # 每一个属性的各种取值情况---输出所有的不同取值
            nvalue = len(unique_value)  # 统计不同值的总个数
            #print(unique_value) #输出列表，类似['青绿' '乌黑' '浅白']
            xi1_vc=pd.value_counts(xi1) #分类统计每个值的个数，返回的是Series，标签为所有可能取值，带名字
            #print(xi1_vc)
            ###################缺失样本补零##########################
            # xi_tem0=np.zeros(nvalue)#产生一个零向量
            
            # xi_tem0=pd.Series(xi_tem0,index=unique_value)#生成一个带标签的一维数组，标签为所有可能的取值
            # print(xi_tem0)
            #xi1_vc=xi1_vc+xi_tem0;#补齐xi1_vc缺的值
            xi1_vc=pd.Series(xi1_vc,unique_value)#补齐xi1_vc缺的值
            #print(xi1_vc)
            nan_exist1=xi1_vc.isnull().values.any()#查看是否有缺失的样本，判断NaN的存在
            if nan_exist1:
            ###################缺失样本补零##########################
                xi1_value_count = xi1_vc.fillna(0) + 1  # 计算正样本中，该属性每个取值的数量，并且加1，即拉普拉斯平滑
                p1_list.append(p_xi(is_continuous, np.log(xi1_value_count / (m1 + nvalue))))
            else:
                p1_list.append(p_xi(is_continuous, np.log(xi1_vc / m1 )))

            xi0_vc=pd.value_counts(xi0) 
            xi0_vc=pd.Series(xi0_vc,unique_value);#补齐xi0_vc缺的值
            nan_exist0=xi0_vc.isnull().values.any()#查看是否有缺失的样本，判断NaN的存在
            if nan_exist0:
                xi0_value_count = xi0_vc.fillna(0) + 1  # 计算负样本中，该属性每个取值的数量，并且加1，即拉普拉斯平滑
                p0_list.append(p_xi(is_continuous, np.log(xi0_value_count / (m0 + nvalue))))#负样本中取对数计算概率
            else:
                p0_list.append(p_xi(is_continuous, np.log(xi0_vc / m0 )))#负样本中取对数计算概率
       

    return p1, p1_list, p0_list


def predict_nb(x, p1, p1_list, p0_list):
    n = len(x)

    x_p1 = np.log(p1)#正样本的概率，对数形式，乘积变成求和
    x_p0 = np.log(1 - p1)#负样本的概率，对数形式
    for i in range(n):
        p1_xi = p1_list[i]
        p0_xi = p0_list[i]

        if p1_xi.is_continuous:
            mean1, var1 = p1_xi.conditional_pro #赋值的形式
            mean0, var0 = p0_xi.conditional_pro 
            #连续型求概率是求积分，在某一点出可用一个单位1长的区间代替，（等价于直接将取值带入概率密度函数）
            x_p1 += np.log(1 / (np.sqrt(2 * np.pi) * var1) * np.exp(- (x[i] - mean1) ** 2 / (2 * var1 ** 2)))
            x_p0 += np.log(1 / (np.sqrt(2 * np.pi) * var0) * np.exp(- (x[i] - mean0) ** 2 / (2 * var0 ** 2)))
        else:
            x_p1 += p1_xi.conditional_pro[x[i]]#两层索引找到当前属性的概率
            
            x_p0 += p0_xi.conditional_pro[x[i]]

    if x_p1 > x_p0:
        return '是'
    else:
        return '否'


if __name__ == '__main__':
    data_path = r'data\watermelon3_0_Ch.csv'
    data = pd.read_csv(data_path, index_col=0)
    #X = data.iloc[:, :-1]
    X = data.iloc[:, :-1] #提取属性
    y = data.iloc[:, -1] #提取类别
    #print(X)
    p1, p1_list, p0_list = train_nb(X, y)
    #print(p1)

    #
    x_test = X.iloc[0, :]   # 书中测1 其实就是第一个数据
    #
    print(predict_nb(x_test, p1, p1_list, p0_list))
   
