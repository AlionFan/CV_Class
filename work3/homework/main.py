import cv2
import numpy as np

def verify_fingerprint(img_path1, img_path2, result_filename=None):
    # 读取图像
    img1 = cv2.imread(img_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Error: 无法读取图像文件")
        return False
    
    # 创建SIFT特征检测器
    sift = cv2.SIFT_create()
    
    # 检测关键点和计算描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        print(f"Error: 无法提取特征描述符")
        return False
    
    # 创建FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # 进行特征匹配
    matches = flann.knnMatch(des1, des2, k=2)
    
    # 应用Lowe's比率测试来筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    
    # 获取匹配点数量
    match_count = len(good_matches)
    print(f"匹配点数量: {match_count}")
    
    # 判断是否验证通过
    is_verified = match_count > 300
    print(f"验证结果: {'通过' if is_verified else '未通过'}")
    
    # 绘制匹配结果
    result_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # 显示匹配结果
    cv2.imshow('指纹匹配结果', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 保存结果图像
    if result_filename:
        cv2.imwrite(f'result/{result_filename}', result_img)
    
    return is_verified

def main():
    # 模板指纹图像路径
    img_path1 = 'homework/pic/fig1.png'
    
    # 测试其他指纹图像
    img_path2 = ['homework/pic/fig2.png', 'homework/pic/fig3.png']
    
    for i, img in enumerate(img_path2, 1):
        print(f"\n正在验证图像: {img}")
        verify_fingerprint(img_path1, img, f'ver{i}.png')

if __name__ == '__main__':
    main()