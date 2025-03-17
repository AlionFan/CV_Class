import cv2
import numpy as np
hg = cv2.imread("./pic/hg.jpg") # ./pic/wyz.jpg
hg_hsv = cv2.cvtColor(hg, cv2.COLOR_BGR2HSV)

min_HSV = np.array([0, 10, 80], dtype="uint8")
max_HSV = np.array([33, 255, 255], dtype="uint8")
mask = cv2.inRange(hg_hsv, min_HSV, max_HSV)
result = cv2.bitwise_and(hg, hg, mask=mask)

cv2.imshow("img", hg)
cv2.imshow("result", result)
cv2.waitKey()
cv2.destroyAllWindows()