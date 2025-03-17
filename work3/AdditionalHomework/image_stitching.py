import cv2
import numpy as np

def find_homography(img1, img2):
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    
    # 检测关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 应用Lowe's ratio测试找到好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    if len(good_matches) < 4:
        return None
    
    # 获取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    return H

def transform_image(img, H, output_shape):
    return cv2.warpPerspective(img, H, (output_shape[1], output_shape[0]))

def stitch_images():
    # 读取输入图像
    img1 = cv2.imread('pic/pic1.jpg')
    img2 = cv2.imread('pic/pic2.jpg')  # 基准图像
    img3 = cv2.imread('pic/pic3.jpg')
    
    if img1 is None or img2 is None or img3 is None:
        print("Error: 无法读取图像文件")
        return
    
    # 获取基准图像的尺寸
    h2, w2 = img2.shape[:2]
    
    # 计算左图到中间图的单应性矩阵
    H1 = find_homography(img1, img2)
    if H1 is None:
        print("Error: 无法找到pic1到pic2的变换")
        return
        
    # 计算右图到中间图的单应性矩阵
    H3 = find_homography(img3, img2)
    if H3 is None:
        print("Error: 无法找到pic3到pic2的变换")
        return
    
    # 计算变换后的图像范围
    h1, w1 = img1.shape[:2]
    h3, w3 = img3.shape[:2]
    
    # 计算左图变换后的边界点
    pts1 = np.float32([[0, 0], [0, h1-1], [w1-1, h1-1], [w1-1, 0]]).reshape(-1, 1, 2)
    dst1 = cv2.perspectiveTransform(pts1, H1)
    
    # 计算右图变换后的边界点
    pts3 = np.float32([[0, 0], [0, h3-1], [w3-1, h3-1], [w3-1, 0]]).reshape(-1, 1, 2)
    dst3 = cv2.perspectiveTransform(pts3, H3)
    
    # 计算全景图的范围
    pts = np.concatenate((dst1, dst3), axis=0)
    min_x = min(min(pts[:, 0, 0]), 0)
    min_y = min(min(pts[:, 0, 1]), 0)
    max_x = max(max(pts[:, 0, 0]), w2)
    max_y = max(max(pts[:, 0, 1]), h2)
    
    # 创建平移矩阵
    move_h = -min_y
    move_w = -min_x
    translation = np.array([[1, 0, move_w], [0, 1, move_h], [0, 0, 1]])
    
    # 计算最终输出图像的大小
    out_h = int(max_y - min_y)
    out_w = int(max_x - min_x)
    
    # 创建输出图像
    output_shape = (out_h, out_w)
    result = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # 变换左图
    H1_translated = translation.dot(H1)
    warped1 = transform_image(img1, H1_translated, output_shape)
    
    # 变换右图
    H3_translated = translation.dot(H3)
    warped3 = transform_image(img3, H3_translated, output_shape)
    
    # 将中间图放在正确的位置
    warped2 = np.zeros_like(result)
    warped2[int(move_h):int(move_h)+h2, int(move_w):int(move_w)+w2] = img2
    
    # 创建掩码
    mask1 = cv2.cvtColor(warped1, cv2.COLOR_BGR2GRAY) > 0
    mask2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY) > 0
    mask3 = cv2.cvtColor(warped3, cv2.COLOR_BGR2GRAY) > 0
    
    # 合并图像
    result = np.zeros_like(warped1)
    result[mask1] = warped1[mask1]
    result[mask2] = warped2[mask2]
    result[mask3] = warped3[mask3]
    
    # 保存结果
    cv2.imwrite('result/result.png', result)
    print('拼接完成！结果已保存为 result/result.png')

if __name__ == '__main__':
    stitch_images() 