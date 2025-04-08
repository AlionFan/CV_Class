# 导入必要的库
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, models, datasets
import time
import copy
from PIL import Image
import json


device = torch.device("cuda")

# 定义数据预处理
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# 加载数据集
data_dir = 'data/flower_data'
image_datasets = {
    'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
    'valid': datasets.ImageFolder(os.path.join(data_dir, 'valid'), data_transforms['valid'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=32, shuffle=True, num_workers=4),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=32, shuffle=False, num_workers=4)
}

# 初始化并调整ResNet18模型
def initialize_model(num_classes, feature_extract, use_pretrained=True):
    # 从本地加载预训练模型
    model_ft = models.resnet18(use_pretrained)
    # 调整最后一层
    num_features = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_features, num_classes)
    model_ft = model_ft.to(device)
    return model_ft

def set_parameter_requires_grad(model, feature_extracting):
    # 设置参数是否需要梯度更新
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

# 训练模型
def train_model(model, dataloaders, criterion, optimizer, num_epochs=2, is_inception=False, filename='./result/3_checkpoint.pth', scheduler=None):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # 每个epoch都有训练和验证阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 梯度清零
                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播 + 优化器更新（仅在训练阶段）
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 保存最佳模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # 保存检查点
                torch.save({
                    'epoch': epoch,
                    'state_dict': best_model_wts,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, filename)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model

# 在模型保存之前，先创建保存目录
# 创建保存模型的目录
save_dir = './result'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 模型保存路径
filename = './result/3_checkpoint.pth'

# 初始化并训练模型（第一阶段）
model_ft = initialize_model(102, feature_extract=True, use_pretrained=True)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 定义优化器和学习率调度器
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=1e-6)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 训练模型（第一阶段）
model_ft = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=2, filename=filename, scheduler=scheduler)

# 加载训练好的模型
try:
    checkpoint = torch.load(filename)
    model_dict = model_ft.state_dict()
    # 过滤出匹配的权重
    pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrained_dict) 
    model_ft.load_state_dict(model_dict)
    print("Successfully loaded checkpoint from", filename)
except FileNotFoundError:
    print(f"No checkpoint found at {filename}, starting from scratch")
    pass

model_ft.to(device)  # 确保模型被移动到了正确的设备

# 解冻所有参数并继续训练（第二阶段）
# 解冻模型所有参数
set_parameter_requires_grad(model_ft, False)

# 定义新的优化器和学习率调度器
optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 训练模型（第二阶段）
model_ft = train_model(model_ft, dataloaders, criterion, optimizer, num_epochs=2, filename=filename, scheduler=scheduler)

# 加载训练好的模型
checkpoint = torch.load(filename)
model_dict = model_ft.state_dict()

# 过滤出匹配的权重
pretrained_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in model_dict and model_dict[k].size() == v.size()}
model_dict.update(pretrained_dict) 
model_ft.load_state_dict(model_dict)

model_ft.to(device)  # 确保模型被移动到了正确的设备

# 加载测试图片并进行预测
def process_image(image_path):
    # 读取测试数据
    img = Image.open(image_path)
    # Resize,thumbnail方法只能进行缩小，所以进行了判断
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop操作
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # 相同的预处理方法
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # 注意颜色通道应该放在第一个位置
    img = img.transpose((2, 0, 1))
    
    return img

# 在测试图片预测部分，添加缺失的代码
def predict(image_path, model, topk=5):
    """使用训练好的模型进行预测"""
    model.eval()
    img = process_image(image_path)
    # 将numpy array转换为torch tensor
    img = torch.from_numpy(img).type(torch.FloatTensor)
    # 增加batch维度
    img = img.unsqueeze(0)
    img = img.to(device)
    
    with torch.no_grad():
        output = model(img)
        # 获取概率
        probabilities = torch.nn.functional.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probabilities, topk)
        
    return top_probs.cpu().numpy()[0], top_indices.cpu().numpy()[0]

# 展示预测结果部分的完善
# 获取一个batch的测试数据
dataiter = iter(dataloaders['valid'])
images, labels = next(dataiter)

# 将数据移动到正确的设备
images = images.to(device)
labels = labels.to(device)

# 将模型设置为评估模式
model_ft.eval()

# 进行预测
with torch.no_grad():
    output = model_ft(images)

# 得到概率最大的预测结果
_, preds_tensor = torch.max(output, 1)
preds = preds_tensor.cpu().numpy()

# 单张图片的预测示例
def predict_single_image(image_path, model, cat_to_name):
    """预测单张图片并返回预测结果"""
    # 获取前k个预测结果
    probs, classes = predict(image_path, model)
    
    # 获取类别名称
    class_names = [cat_to_name[str(cls)] for cls in classes]
    
    return probs, class_names

# 添加模型评估函数
def evaluate_model(model, dataloader):
    """评估模型在测试集上的性能"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

# 在主程序最后添加模型评估
print("\nEvaluating model performance...")
val_accuracy = evaluate_model(model_ft, dataloaders['valid'])
print(f"Validation Accuracy: {val_accuracy:.2f}%")

# 保存完整模型（包括架构）
torch.save({
    'model_state_dict': model_ft.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'class_to_idx': image_datasets['train'].class_to_idx,
}, './result/3_complete_model.pth')

# 示例：预测单张图片
def predict_and_display_image(image_path, model, cat_to_name):
    """预测并显示单张图片的结果"""
    # 预测
    probs, classes = predict_single_image(image_path, model, cat_to_name)
    
    # 显示图片和预测结果
    img = Image.open(image_path)
    fig, (ax1, ax2) = plt.subplots(figsize=(12,5), ncols=2)
    
    # 显示图片
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title('Input Image')
    
    # 显示预测结果
    y_pos = np.arange(len(classes))
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(classes)
    ax2.invert_yaxis()
    ax2.set_title('Top Predictions')
    
    plt.tight_layout()
    plt.savefig("./result/3_single_prediction.png")
    plt.close()

# 如果要测试单张图片，可以使用以下代码：
# test_image_path = 'path_to_your_test_image.jpg'
# predict_and_display_image(test_image_path, model_ft, cat_to_name)

# 加载类别名称
with open('./data/cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

import matplotlib
matplotlib.use('Agg')  # 使用无图形界面的后端

def im_convert(tensor):
    """ 将 PyTorch tensor 转换为可显示的图像格式 """
    # 复制一份以防修改原始数据
    image = tensor.cpu().clone().detach().numpy()
    # 将图像从 CxHxW 转换为 HxWxC 格式
    image = image.transpose(1, 2, 0)
    # 反归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    # 裁剪到 [0,1] 范围
    image = np.clip(image, 0, 1)
    return image

# 获取类别到索引的映射
class_to_idx = image_datasets['train'].class_to_idx
# 创建索引到类别的反向映射
idx_to_class = {v: k for k, v in class_to_idx.items()}

fig = plt.figure(figsize=(20, 20))
columns = 1
rows = 1

ax = fig.add_subplot(rows, columns, 1, xticks=[], yticks=[])
ax.imshow(im_convert(images[0]))
ax.set_title(
    "{} ({})".format(
        cat_to_name[idx_to_class[preds[0]]],  # 使用idx_to_class进行映射
        cat_to_name[idx_to_class[labels[0].item()]]  # 使用idx_to_class进行映射
    ),
    color=("green" if preds[0] == labels[0].item() else "red")
)

# 保存图像
plt.savefig("./result/3_prediction_result.png", bbox_inches='tight', pad_inches=0)

# 关闭 figure，释放资源
plt.close(fig)