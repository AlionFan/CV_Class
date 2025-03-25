import numpy as np
import dlib
import cv2

def main():
	# 打开视频文件
	vs = cv2.VideoCapture("jupyter/video.mp4")

	# 获取视频的第一帧
	ret, first_frame = vs.read()
	if ret:
		# 设置初始追踪区域（使用原始坐标，不进行缩放）
		x, y, w, h = 800, 400, 200, 160
		
		# 获取原始视频尺寸
		(height, width) = first_frame.shape[:2]
		
		# 创建dlib追踪器
		tracker = dlib.correlation_tracker()
		rect = dlib.rectangle(x, y, x + w, y + h)
		rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
		tracker.start_track(rgb, rect)
		
		# 设置输出视频（使用原始尺寸）
		fourcc = cv2.VideoWriter_fourcc(*'mp4v')
		out = cv2.VideoWriter("result/task7/dlib_tracked.mp4", fourcc, 30.0, (width, height))

		# 视频流
		while True:
			# 读取当前帧
			ret, frame = vs.read()
			if not ret:
				break
				
			# 转换颜色空间用于追踪
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			
			# 更新追踪器
			tracker.update(rgb)
			pos = tracker.get_position()
			
			# 获取追踪框坐标
			startX = int(pos.left())
			startY = int(pos.top())
			endX = int(pos.right())
			endY = int(pos.bottom())
			
			# 绘制追踪框
			cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
			cv2.putText(frame, "Tracking", (startX, startY - 10), 
				cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
			
			# 写入输出视频
			out.write(frame)
			
			# 按ESC退出
			if cv2.waitKey(100) & 0xFF == 27:
				break

	# 释放资源
	vs.release()
	out.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()