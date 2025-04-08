import os
import json
import time
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from PIL import Image

from config import param_combinations, base_config


class FlowerClassifier:
    def __init__(self, config):
        """初始化分类器配置"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_directories()
        self.load_data()
        self.initialize_model()
        
    def setup_directories(self):
        """创建必要的目录"""
        os.makedirs(self.config['result_dir'], exist_ok=True)
        
    def load_data(self):
        """加载和预处理数据"""
        # 定义数据预处理
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(self.config['image_size']),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(self.config['image_size'] + 32),
                transforms.CenterCrop(self.config['image_size']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        # 加载数据集
        self.image_datasets = {
            'train': datasets.ImageFolder(
                os.path.join(self.config['data_dir'], 'train'), 
                data_transforms['train']),
            'valid': datasets.ImageFolder(
                os.path.join(self.config['data_dir'], 'valid'), 
                data_transforms['valid'])
        }

        self.dataloaders = {
            'train': torch.utils.data.DataLoader(
                self.image_datasets['train'], 
                batch_size=self.config['batch_size'], 
                shuffle=True, 
                num_workers=4),
            'valid': torch.utils.data.DataLoader(
                self.image_datasets['valid'], 
                batch_size=self.config['batch_size'], 
                shuffle=False, 
                num_workers=4)
        }
        
        # 加载类别名称
        with open(self.config['cat_to_name_path'], 'r') as f:
            self.cat_to_name = json.load(f)
            
        self.class_to_idx = self.image_datasets['train'].class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
    
    def initialize_model(self):
        """初始化模型"""
        # 加载预训练模型
        self.model = models.resnet18(pretrained=self.config['use_pretrained'])
        
        # 冻结或解冻参数
        self.set_parameter_requires_grad(self.model, self.config['feature_extract'])
        
        # 调整最后一层
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(self.class_to_idx))
        self.model = self.model.to(self.device)
        
        # 定义损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 定义优化器
        params_to_update = self.get_params_to_update()
        self.optimizer = optim.Adam(
            params_to_update, 
            lr=self.config['learning_rate']
        )
        
        # 定义学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config['scheduler_step_size'], 
            gamma=self.config['scheduler_gamma']
        )
    
    def set_parameter_requires_grad(self, model, feature_extracting):
        """设置参数是否需要梯度更新"""
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False
    
    def get_params_to_update(self):
        """获取需要更新的参数"""
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
        return params_to_update
    
    def train(self):
        """训练模型"""
        since = time.time()
        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(self.config['num_epochs']):
            print(f'Epoch {epoch}/{self.config["num_epochs"] - 1}')
            print('-' * 10)

            # 每个epoch都有训练和验证阶段
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                running_loss = 0.0
                running_corrects = 0

                # 迭代数据
                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    # 梯度清零
                    self.optimizer.zero_grad()

                    # 前向传播
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = self.model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = self.criterion(outputs, labels)

                        # 反向传播 + 优化器更新（仅在训练阶段）
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # 统计
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    self.scheduler.step()

                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(self.dataloaders[phase].dataset)
                
                # 记录历史数据
                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # 保存最佳模型
                if phase == 'valid' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    self.save_checkpoint(epoch, best_acc)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # 加载最佳模型权重
        self.model.load_state_dict(best_model_wts)
        return history
    
    def save_checkpoint(self, epoch, best_acc):
        """保存模型检查点"""
        checkpoint_path = os.path.join(
            self.config['result_dir'], 
            f'checkpoint_epoch{epoch}.pth'
        )
        
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': best_acc,
            'class_to_idx': self.class_to_idx,
        }, checkpoint_path)
        print(f'Saved checkpoint to {checkpoint_path}')
    
    def evaluate(self):
        """评估模型性能"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.dataloaders['valid']:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Validation Accuracy: {accuracy:.2f}%')
        return accuracy
    
    def predict(self, image_path, topk=5):
        """预测图像类别"""
        self.model.eval()
        img = self.process_image(image_path)
        img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_probs, top_indices = torch.topk(probabilities, topk)
            
        # 转换为类别名称
        top_classes = [self.idx_to_class[idx] for idx in top_indices.cpu().numpy()[0]]
        top_class_names = [self.cat_to_name[cls] for cls in top_classes]
        
        return top_probs.cpu().numpy()[0], top_class_names
    
    def process_image(self, image_path):
        """预处理图像"""
        img = Image.open(image_path)
        
        # 保持宽高比的resize
        width, height = img.size
        if width > height:
            img.thumbnail((10000, 256))
        else:
            img.thumbnail((256, 10000))
            
        # 中心裁剪
        left = (img.width - self.config['image_size']) / 2
        top = (img.height - self.config['image_size']) / 2
        right = left + self.config['image_size']
        bottom = top + self.config['image_size']
        img = img.crop((left, top, right, bottom))
        
        # 转换为numpy数组并归一化
        img = np.array(img) / 255
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # 调整通道顺序
        img = img.transpose((2, 0, 1))
        
        return img
    
    def visualize_predictions(self, num_images=5):
        """可视化预测结果"""
        images, labels = next(iter(self.dataloaders['valid']))
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(images[:num_images])
            _, preds = torch.max(outputs, 1)
        
        fig = plt.figure(figsize=(20, 20))
        for i in range(num_images):
            ax = fig.add_subplot(1, num_images, i+1, xticks=[], yticks=[])
            ax.imshow(self.im_convert(images[i]))
            ax.set_title(
                "{} ({})".format(
                    self.cat_to_name[self.idx_to_class[preds[i].item()]],
                    self.cat_to_name[self.idx_to_class[labels[i].item()]]
                ),
                color=("green" if preds[i] == labels[i] else "red")
            )
        
        save_path = os.path.join(self.config['result_dir'], 'predictions.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    
    def im_convert(self, tensor):
        """转换tensor为可显示的图像格式"""
        image = tensor.cpu().clone().detach().numpy()
        image = image.transpose(1, 2, 0)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
        return image
    
    def plot_training_history(self, history):
        """绘制训练历史图表"""
        plt.figure(figsize=(12, 5))
        
        # 绘制损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # 绘制准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        save_path = os.path.join(self.config['result_dir'], 'training_history.png')
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    # 运行所有参数组合实验
    results = []
    for i, params in enumerate(param_combinations):
        print(f"\nRunning experiment {i+1}/{len(param_combinations)}")
        print(f"Parameters: {params}")
        
        # 合并配置
        current_config = {**base_config, **params}
        
        # 创建并运行分类器
        classifier = FlowerClassifier(current_config)
        history = classifier.train()
        accuracy = classifier.evaluate()
        
        # 保存结果
        results.append({
            'params': params,
            'accuracy': accuracy,
            'history': history
        })
        
        # 为每个实验创建单独的结果目录
        exp_dir = os.path.join(base_config['result_dir'], f'exp_{i+1}')
        os.makedirs(exp_dir, exist_ok=True)
        
        # 保存可视化结果
        classifier.visualize_predictions()
        classifier.plot_training_history(history)
        
        # 保存模型和配置（使用参数命名）
        model_name = f"model_epochs{params['num_epochs']}_lr{params['learning_rate']}_batch{params['batch_size']}.pth"
        torch.save(classifier.model.state_dict(), os.path.join(exp_dir, model_name))
        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            json.dump(current_config, f, indent=2)

    # 打印所有实验结果
    print("\nExperiment Results:")
    for i, result in enumerate(results):
        print(f"Experiment {i+1}:")
        print(f"Parameters: {result['params']}")
        print(f"Validation Accuracy: {result['accuracy']:.2f}%")
        print("-" * 50)
