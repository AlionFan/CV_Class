import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('pic/hg.jpg')

# 创建结构元素
kernel = np.ones((5,5), np.uint8)

# 进行形态学操作
erosion = cv2.erode(img, kernel, iterations=1)  # 腐蚀
dilation = cv2.dilate(img, kernel, iterations=1)  # 膨胀
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # 开运算
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)  # 形态学梯度
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)  # 顶帽
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)  # 黑帽

# 创建图像展示
titles = ['原图', '腐蚀', '膨胀', '开运算', '闭运算', '形态学梯度', '顶帽', '黑帽']
images = [img, erosion, dilation, opening, closing, gradient, tophat, blackhat]

# 设置图像显示
plt.figure(figsize=(16, 8))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
