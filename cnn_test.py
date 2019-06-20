import numpy as np
import matplotlib.pyplot as plt
n = 100 #代表每个种类的个数
m = 3 #种类的个数
k = 2 #维度

X = np.zeros((n*m,k))           #样本的输入
y = np.zeros(n*m,dtype='uint8') #样本的输出
for i in range(m):
    ix = range(n*i,n*(i+1))
    r = np.linspace(0.0,1,n)
    t = np.linspace(i * 4, (i + 1) * 4, n) + np.random.randn(n) * 0.2
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = i
# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()
# print(y)

#需要自己敲定的步长和正则化系数
step_size = 1e-0
reg = 1e-3 #正则化系数

w = 0.01*np.random.randn(k,m)
b = np.zeros((1,m))
num_examples = X.shape[0]
for i in range(200):
    scores = np.dot(X,w)+b       #计算得分

    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    #计算log概率和互熵损失
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples

    # 加上正则化项
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
        print("iteration %d: loss %f" % (i, loss))

     # 计算得分上的梯度
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # 计算和回传梯度
    dW = np.dot(X.T, dscores)
    db = np.sum(dscores, axis=0, keepdims=True)

    dW += reg * W  # 正则化梯度

    # 参数更新
    W += -step_size * dW
    b += -step_size * db
