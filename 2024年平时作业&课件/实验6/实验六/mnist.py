import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # 解决库冲突问题，防止KMP报错

import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置超参数
n_epochs = 3                  # 训练轮数
batch_size_train = 60         # 训练集的批次大小
batch_size_test = 1000        # 测试集的批次大小
learning_rate = 0.01          # 学习率
momentum = 0.5                # 动量，用于SGD优化器
log_interval = 10             # 每10个批次输出一次训练日志
random_seed = 1               # 随机种子，保证实验可复现
torch.manual_seed(random_seed)  # 设置随机种子

# 加载训练数据集
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),  # 转为张量
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))  # 标准化：均值0.1307，标准差0.3081
                               ])),
    batch_size=batch_size_train, shuffle=True)

# 加载测试数据集
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('./data/', train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

# 显示测试集的示例图片
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)  # 获取一批数据

fig = plt.figure()
for i in range(6):  # 显示前6张图片
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')  # 显示灰度图
    plt.title("Ground Truth: {}".format(example_targets[i]))  # 标题为真实标签
    plt.xticks([])
    plt.yticks([])
plt.show()

# 定义神经网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)  # 卷积层1：输入1通道，输出10通道，核大小5x5
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)  # 卷积层2：输入10通道，输出20通道，核大小5x5
        self.conv2_drop = nn.Dropout2d()  # Dropout：随机丢弃部分特征图，防止过拟合
        self.fc1 = None  # 先设置为 None，稍后动态调整
        self.fc2 = nn.Linear(50, 10)   # 全连接层2：输入50，输出10（对应10个分类）

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        if self.fc1 is None:
            # 动态计算展平大小并初始化 fc1
            flattened_size = x.size(1) * x.size(2) * x.size(3)
            self.fc1 = nn.Linear(flattened_size, 50).to(x.device)  # 动态初始化
        x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 初始化网络和优化器
network = Net()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# 记录训练和测试的损失
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]  # 每个epoch结束时的计数

# 定义训练函数
def train(epoch):
    network.train()  # 设置网络为训练模式
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()  # 梯度清零
        output = network(data)  # 前向传播
        loss = F.nll_loss(output, target)  # 计算负对数似然损失
        loss.backward()  # 反向传播
        optimizer.step()  # 参数更新
        if batch_idx % log_interval == 0:  # 每10个批次输出一次日志
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())  # 记录训练损失
            train_counter.append((batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(network.state_dict(), './model.pth')  # 保存网络参数
            torch.save(optimizer.state_dict(), './optimizer.pth')  # 保存优化器参数

# 定义测试函数
def TTT():
    network.eval()  # 设置网络为评估模式
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 关闭梯度计算
        for data, target in test_loader:
            output = network(data)  # 前向传播
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # 计算损失
            pred = output.data.max(1, keepdim=True)[1]  # 获取预测类别
            correct += pred.eq(target.data.view_as(pred)).sum()  # 计算正确个数
    test_loss /= len(test_loader.dataset)  # 平均损失
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# 训练和测试网络
TTT()  # 初始测试
for epoch in range(1, n_epochs + 1):  # 训练指定轮数
    train(epoch)
    TTT()

# 绘制训练和测试的损失曲线
fig = plt.figure()
plt.plot(train_counter, train_losses, color='blue')  # 绘制训练损失
plt.scatter(test_counter, test_losses, color='red')  # 绘制测试损失
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('number of training examples seen')
plt.ylabel('negative log likelihood loss')
plt.show()

# 显示模型预测结果
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
with torch.no_grad():
    output = network(example_data)
fig = plt.figure()
for i in range(6):  # 显示前6个预测结果
    plt.subplot(2, 3, i + 1)
    plt.tight_layout()
    plt.imshow(example_data[i][0], cmap='gray', interpolation='none')  # 显示图像
    plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))  # 显示预测标签
    plt.xticks([])
    plt.yticks([])
plt.show()
