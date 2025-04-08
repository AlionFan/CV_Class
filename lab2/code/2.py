import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 定义超参数
input_size = 28  # 输入尺寸
num_classes = 10  # 类别数量
num_epochs = 3  # 训练周期数
batch_size = 64  # 批次大小

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 加载数据集
train_dataset = datasets.MNIST(
    root='../data',    
    # 数据集存放目录
    train=True,       
    # 使用训练集
    transform=transforms.ToTensor(),
    download=False    
)

test_dataset = datasets.MNIST(
    root='../data',
    train=False,
    transform=transforms.ToTensor(),
    download=False
)

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# 定义CNN模型
class CNN(nn.Module):  # 定义CNN类
    def __init__(self):
        # 初始化层（卷积层、激活函数、池化层、全连接层）
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        # self.pool3 = nn.MaxPool2d(kernel_size=2, padding=1)

        self.fc = nn.Linear(64 * 7 * 7, 10)
    
    def forward(self, x):
        # 前向传播
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        # x = self.pool3(x)

        x = x.view(x.size(0), -1)  # 展平操作
        x = self.fc(x)
        return x

# 定义准确率计算函数
def accuracy(predictions, labels):
    # 计算准确率
    return (predictions.argmax(dim=1) == labels).float().mean()

# 实例化模型、损失函数和优化器
net = CNN().to(device)
criterion = nn.CrossEntropyLoss()  # 定义损失函数
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 定义优化器

# 训练循环
for epoch in range(num_epochs):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到正确的设备上
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')
    
    # 每个epoch结束后评估模型
    net.eval()
    test_loss = 0
    test_accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += criterion(output, target).item()
            test_accuracy += accuracy(output, target).item()
    
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader)
    print(f'Epoch {epoch}: Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

# 保存模型
torch.save(net.state_dict(), './result/2_model.pth')


