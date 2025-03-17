import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('pic/hg.jpg', 0)

# 创建子图
plt.figure(figsize=(15, 10))

# 原图
plt.subplot(231)
plt.imshow(img, cmap='gray')
plt.title('原始图像')

# Sobel算子边缘检测
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
sobel = np.sqrt(sobelx**2 + sobely**2)
plt.subplot(232)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel边缘检测')

# Scharr算子边缘检测
scharrx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=-1)  # X方向梯度
scharry = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=-1)  # Y方向梯度
scharr = np.sqrt(scharrx**2 + scharry**2)  # 梯度幅值计算
plt.subplot(233)
plt.imshow(scharr, cmap='gray')
plt.title('Scharr边缘检测')

# Laplacian算子边缘检测
laplacian = cv2.Laplacian(img, cv2.CV_64F)
plt.subplot(234)
plt.imshow(np.absolute(laplacian), cmap='gray')
plt.title('Laplacian边缘检测')

# Canny边缘检测
canny = cv2.Canny(img, 100, 200)
plt.subplot(235)
plt.imshow(canny, cmap='gray')
plt.title('Canny边缘检测')

plt.tight_layout()
plt.savefig('Figure_4.png')
plt.show()
