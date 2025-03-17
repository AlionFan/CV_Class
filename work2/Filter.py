import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('pic/hg.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 均值滤波
mean_blur = cv2.blur(img, (5,5))

# 高斯滤波
gaussian_blur = cv2.GaussianBlur(img, (5,5), 0)

# 中值滤波
median_blur = cv2.medianBlur(img, 5)

# 双边滤波
bilateral_blur = cv2.bilateralFilter(img, 9, 75, 75)

# 创建图像对比展示
plt.figure(figsize=(12,8))

plt.subplot(231)
plt.imshow(img)
plt.title('原图')
plt.axis('off')

plt.subplot(232)
plt.imshow(mean_blur)
plt.title('均值滤波')
plt.axis('off')

plt.subplot(233)
plt.imshow(gaussian_blur)
plt.title('高斯滤波')
plt.axis('off')

plt.subplot(234)
plt.imshow(median_blur)
plt.title('中值滤波')
plt.axis('off')

plt.subplot(235)
plt.imshow(bilateral_blur)
plt.title('双边滤波')
plt.axis('off')

plt.tight_layout()
plt.savefig('Figure_2.png')
plt.show()

