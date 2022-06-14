#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli
import networkx as nx
from scipy.linalg import block_diag#块矩阵拼接需要用到block_diag命令


# In[2]:


data_m=1000                   #每个节点数据集的个数
N = 3                        #x的维数
n = NETWORK_SIZE = 100       #节点的个数
xi=np.array([0,0.5,1])       #vim用到的x值，是固定的.
PROBABILITY_OF_EAGE = 0.5    #随机图的概率参数
iter0=1500


# In[3]:


###生成系数数据 uim，data_uim是一个多维数组，列表中每一个元素是一个1000*3的矩阵。
data_uim = np.zeros((n,data_m,N))
i=j=k=u=0
for i in range(n):
    np.random.seed(i) #注意添加np.，否则seed可能无效。
    uim=np.random.normal(0,10,size=(data_m,N))#生成一个100*3的多位数组，均值为0，标准差为10。
    uim[:,N-1]=1#将第三列改为1，论文设定。
    data_uim[i,:,:]=uim
###生成数据 vim，data_vim是一个三维数组，列表中每一个元素是一个1*1000的矩阵。
data_vim=np.zeros((n,1,data_m))
sum_vim=np.zeros((1,data_m))
vim = np.zeros((1,data_m))
p1= np.zeros((1,data_m))
for j in range(n):
    for k in range(data_m):
        p1[0,k]=1/(1+np.exp(np.dot(xi,np.transpose(data_uim[j,k,:]))))
    for u in range(data_m):
        np.random.seed(u+n)
        sum_vim[:,u] = bernoulli.rvs(size=1,p=p1[0,u])#target;每个节点中的每个数据，所用的p是变量，只能使用循环。
    data_vim[j,:,:] = sum_vim


# In[4]:


k=20
def generateRandomNetwork():
    # 生成邻接矩阵
    for i in range(NETWORK_SIZE):
        for j in range(i-k,i):
            if j<0:
                a=n+j
            else:
                a=j
            MAT_Adjacent[a][i] = MAT_Adjacent[i][a] = 1
        for u in range(i+1,i+1+k):
            if u>(n-1):
                b=u-n
            else:
                 b=u   
            MAT_Adjacent[b][i] = MAT_Adjacent[i][b] = 1

    # 生成度矩阵，并验证算法是否正确
    degree = MAT_Adjacent.sum(axis=0)  # 计算lie和，degree用于存放每个节点的度
    for i in range(NETWORK_SIZE):  # 计算度矩阵
        MAT_Degree[i][i] += degree[i]
    average_degree = degree.sum() / NETWORK_SIZE
    print('平均度为' + str(average_degree))  # 计算平均度，所有节点度的和除以节点数
    
    identify = 0.0
    statistic = np.zeros((NETWORK_SIZE), dtype=float)  # statistic将用于存放度分布的数组，数组下标为度的大小，对应数组内容为该度的概率
    for i in range(NETWORK_SIZE):
        statistic[degree[i]] = statistic[degree[i]] + 1
    for i in range(NETWORK_SIZE):
        statistic[i] = statistic[i] / NETWORK_SIZE
        identify = identify + statistic[i]
    identify = int(identify)  # 取整
    print('如果output为1则该算法正确\toutput=' + str(identify))  # 用于测试算法是否正确

    # 生成网络双随机权重矩阵，基于拉普拉斯方法
    degree_max = 0
    for i in range(NETWORK_SIZE):
        if degree_max < MAT_Degree[i][i]:
            degree_max = MAT_Degree[i][i]
    Alpha = 1 / degree_max
    for i in range(NETWORK_SIZE):
        for j in range(NETWORK_SIZE):
            if (i == j):
                MAT_EdegWight[i][j] = 1 - MAT_Degree[i][i] * Alpha
            elif (MAT_Adjacent[i][j] == 1):
                MAT_EdegWight[i][j] = Alpha

# 将ER网络写入文件中
def writeRandomNetworkToFile():
    ARRS = []  # 创建字典型变量，用于之后检测问题
    f = open('randomNetwork.txt', 'w+')  # 用写的方式打开本地文件，若文件不存在，则自动建立新文件

    f.write('The adjacent matrix is:\n')
    blank_num = 2  # 规范化输出
    t = NETWORK_SIZE
    while (t >= 10):
        t /= 10
        blank_num += 1
    num_end_position = 1
    for i in range(NETWORK_SIZE):
        f.write(str(i + 1))
        if (i+1)%(10**num_end_position) == 0:
            num_end_position += 1
        for j in range(blank_num+1-num_end_position):
            f.write(' ')
        t = MAT_Adjacent[i]  # 邻接矩阵对应数值
        ARRS.append(t)
        for j in range(NETWORK_SIZE):
            s = str(t[j])  # 强制转换成字符串形式
            f.write(s)
            f.write(' ')
        f.write('\n')  # 一行结束，进行下一行填写

    # f = open('randomNetwork.txt', 'w+')  # 将度分布写入文件名为degree01文件中，若磁盘中无此文件将自动新建
    f.write('\nThe degree matrix is:\n')
    blank_num = 2  # 规范化输出
    t = NETWORK_SIZE
    while (t >= 10):
        t /= 10
        blank_num += 1

    ARRS = []
    num_end_position = 1
    for i in range(NETWORK_SIZE):
        f.write(str(i + 1))
        if (i + 1) % (10 ** num_end_position) == 0:
            num_end_position += 1
        for j in range(blank_num + 1 - num_end_position):
            f.write(' ')
        t = MAT_Degree[i]  # 邻接矩阵对应数值
        ARRS.append(t)
        for k in range(NETWORK_SIZE):
            s = str(t[k])  # 强制转换成字符串形式
            f.write(s)
            f.write(' ')
        f.write('\n')  # 一行结束，进行下一行填写
    num_end_position = 1

    f.write('\nThe distribution probability of degree is:\n')
    degree = MAT_Adjacent.sum(axis=1)  # 计算行和，degree用于存放每个节点的度
    statistic = np.zeros((NETWORK_SIZE), dtype=float)  # statistic将用于存放度分布的数组，数组下标为度的大小，对应数组内容为该度的概率
    for i in range(NETWORK_SIZE):
        statistic[degree[i]] = statistic[degree[i]] + 1
    for i in range(NETWORK_SIZE):
        statistic[i] = statistic[i] / NETWORK_SIZE
    for i in range(NETWORK_SIZE):
        f.write(str(i + 1))
        if (i + 1) % (10 ** num_end_position) == 0:
            num_end_position += 1
        for j in range(blank_num + 1 - num_end_position):
            f.write(' ')
        s = str(statistic[i])  # 注意写入操作要求是字符串格式，因此用str进行格式转换
        f.write(str(s))  # 写入的每一行由两部分组成，一个元素为度的下标，第二个元素为度的概率
        f.write('\n')  # 每个节点的度及概率写入完成将进行换行，输入下一个节点的度及度分布
    f.close()

# 用于绘制ER图
def showGraph():
    G = nx.Graph()
    nodes = np.array(range(NETWORK_SIZE))
    G.add_nodes_from(nodes)
    for i in range(len(MAT_Adjacent)):
        for j in range(len(MAT_Adjacent)):
            if MAT_Adjacent[i][j] == 1:  # 如果不加这句将生成完全图，ER网络的邻接矩阵将不其作用
                G.add_edge(i, j)
    position = nx.circular_layout(G)
    nx.draw_networkx_nodes(G, position, nodelist=nodes, node_color="r")
    nx.draw_networkx_edges(G, position)
    nx.draw_networkx_labels(G, position)
    plt.show()


# In[5]:


MAT_Adjacent = np.zeros((NETWORK_SIZE, NETWORK_SIZE), dtype=int)  # 初始化邻接矩阵
MAT_Degree = np.zeros((NETWORK_SIZE, NETWORK_SIZE), dtype=int)  # 初始化度矩阵
MAT_Laplacian = np.zeros((NETWORK_SIZE, NETWORK_SIZE), dtype=int)  # 初始化拉普拉斯矩阵（不一定用到）
MAT_EdegWight = np.zeros((NETWORK_SIZE, NETWORK_SIZE), dtype=float)  # 初始化边权重阵

generateRandomNetwork()  # 生成ER随机网络
writeRandomNetworkToFile()  # 将随机网络写入randomNetwork.txt文件中
showGraph()
W=MAT_EdegWight
D=MAT_Degree
A=MAT_Adjacent#邻接矩阵
L=D-A


# In[6]:


def loss_1(x):
    data_loss=np.zeros(n)#定义为数组而不是列表会减少很多麻烦
    i=j=0
    for i in range(n):
        loss = 1/data_m*(np.log(1+np.exp(np.dot(x,np.transpose(data_uim[i,:,:]))-data_vim[i,:,:]*np.dot(x,np.transpose(data_uim[i,:,:])))))
        #for j in range(data_m):
            #loss[j]=1/data_m*(math.log(1+math.exp(np.dot(x,np.transpose(data_uim[i,j,:]))-data_vim[i,:,j])))
        loss = sum(loss)
        loss = sum(loss)
        data_loss[i]= loss
    sum_loss=sum(data_loss)
    return sum_loss 


# In[7]:


def grad_1(x):
    data_grad=np.zeros((n,3))
    i=j=0
    for i in range(n):
        grad = np.dot((1-data_vim[i,:,:]-1/(1+np.exp(np.dot(x,np.transpose(data_uim[i,:,:]))))),data_uim[i,:,:])
        data_grad[i,:]= grad
        #for j in range(data_m):
            #grad = 1/(1+np.exp(np.dot(x,np.transpose(data_uim[i,j,:]))))*math.exp(np.dot(x,np.transpose(data_uim[i,j,:])))*data_uim[i,j,:]
            #data_grad[i,j,:]=grad
    data_grad=sum(data_grad)#行相加。
    data_grad=1/data_m*data_grad
    return data_grad


# In[8]:


###利用集中式梯度下降求解f的近似最优值
if __name__ == '__main__':
    iter1=10000#迭代次数
    x1=np.array([0,0,0])#第一次迭代的初值
    r=0.0006#步长
    plt_loss_0=[]#记录loss值。
    for i in range(iter1):
        loss=loss_1(x1)
        plt_loss_0.append(loss)
        grad=grad_1(x1)
        x1=x1-grad*r#梯度下降算法
    x_0=x1
    GD_loss=loss#最小的loss
    loss_x =np.array(range(iter1))   
    loss_y =np.array(plt_loss_0)
    plt.plot(loss_x, loss_y)
    plt.yscale('log')#纵坐标以对数形式显示。
    plt.show()
    print('利用集中式算法得到的最优x为',x1)
    print('最后一次迭代的梯度为',sum(grad))
    print('利用集中式算法得到的最低loss为',GD_loss)


# In[9]:


def loss_2(x):
    #data_loss=np.zeros(n)#定义为数组而不是列表会减少很多麻烦
    loss=0
    sum_loss=0
    for i in range(n):
        loss=sum((x[i]-x_0)**2)
        sum_loss=loss+sum_loss
        #loss=1/data_m*(np.dot(x_ave,np.transpose(data_uim[i]))-data_vim[i])**2        
        #loss=sum(loss)
        #loss=sum(loss)
        #data_loss[i]=loss
    #DNGD_loss=sum(data_loss)
    return sum_loss


# In[10]:


#分布式的梯度函数1
def grad_2(x):
    data_grad=np.zeros((n,3))
    i=j=0
    for i in range(n):
        grad = np.dot((1-data_vim[i,:,:]-1/(1+np.exp(np.dot(x[i,:],np.transpose(data_uim[i,:,:]))))),data_uim[i,:,:])
        data_grad[i,:]= grad
        #for j in range(data_m):
            #grad = 1/(1+np.exp(np.dot(x,np.transpose(data_uim[i,j,:]))))*math.exp(np.dot(x,np.transpose(data_uim[i,j,:])))*data_uim[i,j,:]
            #data_grad[i,j,:]=grad
    data_grad=1/data_m*data_grad
    return data_grad


# In[11]:


#分布式的二阶导
def grad_22(x):
    data=[]
    xs=np.zeros((data_m,N))
    for i in range(data_m):
        xs[i]=x
    for k in range(n):
        uim=data_uim[k]
        block1=np.dot(np.transpose(uim),uim)
        xishu=np.exp(np.dot(np.transpose(uim),xs))/((1+np.exp(np.dot(np.transpose(uim),xs)))**2)
        data.append(1/data_m*xishu*block1)
    a0=block_diag(data[0],data[1])#命令表示，大矩阵的每个对角线元素是个小矩阵
    for i in range(n-2):
        a0=block_diag(a0,data[i+2])
    return a0


# In[12]:


###算法1，使用邻居状态
if __name__ == '__main__':
    x=np.ones((n,N))
    loss1=loss_2(x)#画图时y轴的分母
    a1=0.05
    loss=0
    grad=0
    plt_loss_1=[]
    for i in range(iter0):
        a=a1/(i+1)
        grad=grad_2(x)
        x=np.dot(W,x)-a*grad
        loss=loss_2(x)
        loss=loss
        plt_loss_1.append(loss)
    algorithm_1_loss=loss
    plt.plot(range(iter0),np.array(plt_loss_1/loss1))#默认98.2434为最优值。f-f*,f*默认为GD算法的最小loss
    plt.yscale('log')
    plt.show()
    print(x[1,:])
    print('最后一次迭代的梯度为',sum(grad))
    print('利用算法得到的最低loss为',algorithm_1_loss)


# In[13]:


###EXTRA
if __name__ == '__main__':
    x=np.ones((n,N))
    a=0.05
    x_1=np.dot(W,x)-a*grad_2(x)
    W_=(np.eye(n,n)+W)/2
    plt_loss_2=[]
    for i in range(iter0):
        x_2=np.dot(np.eye(n,n)+W,x_1)-np.dot(W_,x)-a*(grad_2(x_1)-grad_2(x))#EXTRA
        x=x_1
        x_1=x_2
        loss=loss_2(x)
        plt_loss_2.append(loss)
    print('利用算法得到的最低loss为',loss)
    plt.plot(np.array(range(iter0)),np.array(plt_loss_2/loss1))
    print(x[0])
    plt.yscale('log')
    plt.show()


# In[ ]:





# In[14]:


###D-NG
if __name__ == '__main__':
    loss=0
    x=y=np.ones((n,N))
    a=0.058
    p=0
    plt_loss_3=[]
    for i in range(iter0):
        a_1=a/(1+i)
        p=i/(i+3)
        x_1=x#D-NG
        x=np.dot(W,y)-a_1*grad_2(y)
        y=x+p*(x-x_1)
        loss=loss_2(x)
        plt_loss_3.append(loss)
    print('利用算法得到的最低loss为',loss)
    plt.plot(np.array(range(iter0)),np.array(plt_loss_3/loss1))
    print(x[0])
    plt.yscale('log')
    plt.show()


# In[ ]:





# In[15]:


###Acc-DNGD-SC
if __name__ == '__main__':
    x=v=y=np.ones((n,N))
    loss=0
    sloss=loss_2(x)/n
    plt_loss_4=[]
    s=grad_2(y)
    a=0.04
    b=0.005
    for i in range(iter0):
        x=np.dot(W,y)-b*s#Acc-DNGD-SC
        v=(1-a)*np.dot(W,v)+a*np.dot(W,y)-b/a*s
        grad0=grad_2(y)
        y=(x+a*v)/(1+a)
        s=np.dot(W,s)+grad_2(y)-grad0
        loss=loss_2(x)
        plt_loss_4.append(loss)
    print('利用算法得到的最低loss为',loss)
    loss_x =np.array(range(iter0))
    loss_y =np.array(plt_loss_4/loss1)
    plt.plot(loss_x,loss_y)
    print(x[0])
    plt.yscale('log')
    plt.show()


# In[ ]:





# In[16]:


#根据相关矩阵的特征根计算系数
c=0.1
L1=np.kron(L,np.eye(N))
grad2=grad_22(x_0)
L2=np.dot(np.eye(n*N)-c*L1,grad2)
L3=L1+L2
for i in range(np.shape(L3)[0]):#将数据为nan的，转为0
    for j in range(np.shape(L3)[1]):
        if np.isnan(L3[i][j]):
             L3[i][j] = 0
eigenvalue, featurevector=np.linalg.eig(L3)
max_eig=max(filter(lambda x: x > 0, eigenvalue))
min_eig=min(filter(lambda x: x > 0, eigenvalue))
alpha=(2/(max_eig**0.5+min_eig**0.5))**2
theta1=((max_eig**0.5-min_eig**0.5)/(max_eig**0.5+min_eig**0.5))**2
print(alpha)
print(theta1)
if __name__ == '__main__':
    x0=x1=np.zeros((n,N))
    x2=np.ones((n,N))
    loss=0
    grad=0
    plt_loss_5=[]
    for k in range(iter0):
        x0=x1#存k-1时刻的值
        grad0=grad_2(x0)
        x1=x2#存k时刻的值
        grad1=grad_2(x1)
        x2=np.dot((np.eye(n)-alpha*L),x1)-alpha*np.dot((np.eye(n)-c*L),grad1)+theta1*(x1-x0)
        loss=loss_2(x2)
        plt_loss_5.append(loss)
    plt.plot(range(iter0),np.array(plt_loss_5/loss1))#默认98.2434为最优值。f-f*,f*默认为GD算法的最小loss
    plt.yscale('log')
    plt.show()
    print(x2[0,:])
    print('最后一次迭代的梯度为',sum(grad1))
    print('利用算法得到的最低loss为',loss)


# In[ ]:





# In[17]:


plt.title(r'(b) $k$-cycle graph')
x_axix=np.array(range(iter0))
plt.plot(x_axix[0:iter0:120],np.array(plt_loss_1/loss1)[0:iter0:120], color='green', label='DGD',marker='s')
plt.plot(x_axix[0:iter0:120],np.array(plt_loss_2/loss1)[0:iter0:120], color='m', label='EXTRA',marker='o')
plt.plot(x_axix[0:iter0:120],np.array(plt_loss_3/loss1)[0:iter0:120], color='orange', label='D-NG',marker='d')
plt.plot(x_axix[0:iter0:120],np.array(plt_loss_4/loss1)[0:iter0:120], color='blue', label='Acc-DNGD-SC',marker='^')
plt.plot(x_axix[0:iter0:120],np.array(plt_loss_5/loss1)[0:iter0:120], color='red', label='DHB',marker='*')
plt.legend() # 显示图例
plt.xlabel(r'$k$')
plt.ylabel('Residual')
plt.yscale('log')
plt.savefig('(b) k-cycle graph（2）.eps', dpi=500, bbox_inches='tight')
plt.show()


# In[ ]:




