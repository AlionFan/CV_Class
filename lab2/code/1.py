import numpy as np
import torch
from torch import nn
from d2l import torch as d2l

# 任务：多通道卷积计算

import numpy as np

"""
    执行多通道2D卷积计算（带bias）
    参数:
        input_matrix: 输入矩阵 (channels, height, width)
        kernel: 卷积核 (out_channels, in_channels, kernel_height, kernel_width)
        bias: 偏置 (out_channels,)，默认为None
        stride: 步幅，默认为1
        padding: 填充，默认为0
    返回:
        output: 卷积结果 (out_channels, output_height, output_width)
"""
def convolution_multi_channel(input_matrix, kernel, bias=None, stride=1, padding=0):
        
        # 转换为numpy数组
        input_matrix = np.array(input_matrix)
        kernel = np.array(kernel)
        
        # 获取维度信息
        if len(input_matrix.shape) == 2:  # 单通道情况
            input_matrix = input_matrix[np.newaxis, :, :]
        in_channels, input_h, input_w = input_matrix.shape
        
        if len(kernel.shape) == 2:  # 单输入单输出情况
            kernel = kernel[np.newaxis, np.newaxis, :, :]
        out_channels, kernel_in_channels, kernel_h, kernel_w = kernel.shape
        
        # 验证输入通道数是否匹配
        if in_channels != kernel_in_channels:
            raise ValueError("输入矩阵的通道数必须与kernel的输入通道数匹配")
        
        # 处理bias
        if bias is None:
            bias = np.zeros(out_channels)
        
        # 添加padding
        input_matrix = np.pad(input_matrix, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
        # 计算输出尺寸
        output_h = (input_h - kernel_h + 2 * padding) // stride + 1
        output_w = (input_w - kernel_w + 2 * padding) // stride + 1
        
        # 初始化输出矩阵
        output = np.zeros((out_channels, output_h, output_w))
        
        # 执行卷积操作
        for out_channel in range(out_channels):
            for h in range(output_h):
                for w in range(output_w):
                    output[out_channel, h, w] = np.sum(
                        input_matrix[:, h*stride:h*stride+kernel_h, w*stride:w*stride+kernel_w] * kernel[out_channel, :, :, :]
                    ) + bias[out_channel]
        
        return output


# 示例2：多通道输入
input_multi = np.array([
        [[1, 1, 2, 2, 1],    # 通道1
         [1, 1, 1, 2, 1],
         [2, 1, 1, 0, 2],
         [2, 1, 0, 1, 2],
         [2, 1, 2, 2, 2]],
        [[0, 1, 2, 0, 1],    # 通道2
         [2, 2, 1, 1, 0],
         [1, 0, 0, 0, 2],
         [0, 1, 0, 1, 2],
         [0, 1, 0, 1, 2]],
         [[2, 2, 0, 1, 2],
          [0, 0, 2, 1, 2],
          [2, 1, 0, 2, 1],
          [1, 1, 0, 0, 0],
          [0, 0, 1, 1, 1]]
    ])
kernel_multi = np.array([
        [              # 输出通道1
            [[1, 1, 1],   # 输入通道1的kernel
             [-1, -1, 0],
             [-1, 1, 0]],
            [[-1, -1, 1],   # 输入通道2的kernel
             [-1, 1, 0],
             [-1, 1, 0]],
             [[1, 0, -1],   # 输入通道3的kernel
              [0, 0, 0],
              [1, -1, -1]]
        ],
        [              # 输出通道2
            [[0, 0, -1],   # 输入通道1的kernel
             [-1, 1, 1],
             [0, 0, 0]],
            [[0, 0, 1],   # 输入通道2的kernel
             [1, 0, 1],
             [0, -1, -1]],
             [[-1, 1, 1],   # 输入通道3的kernel
              [0, 1, 1],
              [1, -1, 1]]
        ]
    ])
bias = np.array([1, 0])  # 为每个输出通道定义bias
    
print("\n多通道结果:")
result2 = convolution_multi_channel(input_multi, kernel_multi, bias, stride=2, padding=1)
print("输出通道1:")
print(result2[0])
print("输出通道2:")
print(result2[1])