import cv2
import numpy as np
from PIL import Image

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
data_dir = './data/images/'
mask_dir = './data/mask/'


img = cv2.imread(mask_dir+'mask_3dfa40a4d772281fa2211a73cab7c2e.jpg',cv2.IMREAD_GRAYSCALE)


# 高斯滤波
blurred = cv2.GaussianBlur(img, (9, 9), 0)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Canny边缘检测
edges = cv2.Canny(blurred, 50, 150)

cv2.imshow('Contours', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
print(len(contours))
cv2.drawContours(img_rgb, contours, -1, (0, 255, 0), 2)

# 计算轮廓边缘方向
for contour in contours:
    print(cv2.arcLength(contour, True))
    for i in range(len(contour) - 1):
        point1 = contour[i][0]
        point2 = contour[i + 1][0]
        direction = np.arctan2(point2[1] - point1[1], point2[0] - point1[0]) * 180 / np.pi
        # 可以根据需要对方向进行进一步处理或统计

cv2.imshow('Contours', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

if len(contours) > 0:
    epsilon = 1 * cv2.arcLength(contours[0], True)
    approx = cv2.approxPolyDP(contours[0], epsilon, True)
    # 创建一个空白图像用于绘制多边形
    poly_image = np.zeros_like(img_rgb)
    cv2.drawContours(poly_image, [approx], 0, (0, 255, 0), 2)
    cv2.imshow("Polygon", poly_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 计算边缘方向
gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=4)
cv2.imshow('Contours', gradient_x)
cv2.waitKey(0)
cv2.destroyAllWindows()
gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
gradient_direction = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)
print(gradient_direction)

# # 显示结果
# plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title('Original Image')
# plt.subplot(1, 2, 2), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
# plt.show()