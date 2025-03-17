import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('pic/hg.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建子图布局
plt.figure(figsize=(12, 8))

# 显示原始灰度图
plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('原始灰度图')

# 计算并显示直方图
plt.subplot(2, 2, 2)
plt.hist(gray.ravel(), 256, [0, 256])
plt.title('灰度直方图')

# 直方图均衡化
equ = cv2.equalizeHist(gray)
plt.subplot(2, 2, 3)
plt.imshow(equ, cmap='gray')
plt.title('直方图均衡化结果')

# 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)
plt.subplot(2, 2, 4)
plt.imshow(cl1, cmap='gray')
plt.title('自适应直方图均衡化结果')

plt.tight_layout()
plt.savefig('Figure_3.png')
plt.show()
