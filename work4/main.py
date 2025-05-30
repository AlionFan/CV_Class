import cv2
import numpy as np
import os

# 检查视频文件是否存在
video_path = "video/video1.mov"
if not os.path.exists(video_path):
    print(f"错误：视频文件 {video_path} 不存在")
    exit()

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 检查视频是否成功打开
if not cap.isOpened():
    print(f"错误：无法打开视频文件 {video_path}")
    exit()

# 获取视频的基本信息
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"视频信息：")
print(f"帧率: {fps}")
print(f"总帧数: {frame_count}")
print(f"分辨率: {width}x{height}")

# 读取第一帧
ret, frame = cap.read()
if not ret:
    print("错误：无法读取视频第一帧")
    cap.release()
    exit()

# 创建视频写入器
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('./result/video1.mp4', fourcc, fps, (width, height))

c, r, w, h = 800, 400, 200, 160
trace_window = (c, r, w, h)
roi = frame[r:r+h, c:c+w]

hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([hsv_roi], [0], None, [180], [0, 180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# 设置终止条件
termcriteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 10, 0.03)

frame_number = 0
while True:
    # 读取一帧
    ret, frame = cap.read()
    frame_number += 1
    
    # 如果读取失败，退出循环
    if not ret:
        print(f"警告：无法读取第 {frame_number} 帧，视频可能已结束")
        break
    
    # 显示图像
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # meanShift 和 CamShift
    # ret, track_window = cv2.meanShift(dst, trace_window, termcriteria)
    ret, track_window = cv2.CamShift(dst, trace_window, termcriteria)
    
    x, y, w, h = track_window
    img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
    
    # 保存当前帧到视频文件
    out.write(img2)
    
    # 显示图像
    cv2.imshow('Video', img2)
    
    # 按'q'键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
out.release()
cv2.destroyAllWindows()

