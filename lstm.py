import pandas as pd
import numpy as np
import csv
from torch import nn
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

data_csv = pd.read_csv("./okx_orderbook.csv")
data_csv = data_csv.drop(data_csv.columns[[0,1, -1]], axis=1)

dataset = data_csv.values
def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i: (i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back][0])
    return np.array(dataX), np.array(dataY)

dataX, dataY = create_dataset(dataset, 2)
train_size = int(len(dataX) * 0.7)
test_size = len(dataX) - train_size
train_X = torch.from_numpy(dataX[:train_size]).float()
train_Y = torch.from_numpy(dataY[:train_size]).float()
test_X = torch.from_numpy(dataX[train_size:]).float()
test_Y = torch.from_numpy(dataY[train_size:]).float()

feature = train_X.shape[2] # 20喂数据
seq_len = train_X.shape[1] # 利用过去两次的bid/ask px 和 size作为输入序列
hidden_size = 64 # 隐藏层
num_layers = 1
output_size = 1 # 输出维度 输出输出的预测的price

lr = 1e-1 #学习率
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# 定义模型
class lstm_reg(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = 1  # 单向LSTM
        #         self.batch_size = batch_size
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size).to(device)
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(x, (h_0, c_0))  # output
        #         print("output is",output.shape)
        #         print(output[0])

        pred = self.linear(output)
        #         print("pred is",pred.shape)
        #         print(pred[0])

        pred = pred[:, -1, :]
        #         print("pred2 is",pred.shape)
        #         print(pred[0])

        return pred

net = lstm_reg(feature, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

epochs = 100000
for e in range(epochs):
    var_x = Variable(train_X)
    var_y = Variable(train_Y)
    # 前向传播
    out = net(var_x)

    # 反向传播
    optimizer.zero_grad()
    loss = criterion(out, var_y)
    loss.backward()
    optimizer.step()
    if (e + 1) % 100 == 0: # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
#         print("pre is",out[0])
#         print(out[0])
#         print("y is", var_y[0])
#         print(var_y[0])


net = net.eval()  # 转换成测试模式

var_data = Variable(test_X)
pred_test = net(var_data)  # 测试集的预测结果

# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()
# pred_test
# # 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(test_Y, 'b', label='real')
plt.legend(loc='best')
plt.show()
