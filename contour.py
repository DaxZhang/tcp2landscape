import cv2
import numpy as np
from PIL import Image

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# 读取图像并转换为灰度图
data_dir = './data/images/'
mask_dir = './data/mask/'



def get_contour(filename, debug = False):
    img = cv2.imread(mask_dir+filename,cv2.IMREAD_GRAYSCALE)

    # 高斯滤波
    blurred = cv2.GaussianBlur(img, (9, 9), 0)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    contours, hierarchy = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



    # 绘制轮廓
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
        epsilon = 0.02 * cv2.arcLength(contours[0], True)
        approx = cv2.approxPolyDP(contours[0], epsilon, True)
        
        approx = np.array(approx).squeeze(axis=1)
        
        if debug:
            # 创建一个空白图像用于绘制多边形
            poly_image = np.zeros_like(img_rgb)
            cv2.drawContours(poly_image, [approx], 0, (0, 255, 0), 2)
            cv2.imshow("Polygon", poly_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        

        return approx





if __name__ == '__main__':

    filename = 'mask_shan1.png'
    get_contour(filename,debug= True)