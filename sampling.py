import cv2
from PIL import Image, ImageDraw
import numpy as np
import math


# 定义分段函数，这里简单示例一个包含三角函数和指数函数的分段情况
# 可以根据你的实际需求调整这个函数的逻辑
def piecewise_function(x):
    if x < 0.5:
        return 100 * np.sin(2 * np.pi * x)
    else:
        return 100 * np.exp(-2 * x)

# 设置图片的宽度和高度
width = 800
height = 600
# 创建一个白色背景的PIL图像对象
image = Image.new('RGB', (width, height), 'white')
draw = ImageDraw.Draw(image)

# 采样点的数量，你可以调整这个数量来改变折线的精细程度
num_samples = 100
# 计算采样点的x坐标间隔



x_step = width / num_samples

# 用于存储折线的坐标点列表
points = []
for i in range(num_samples):
    x = i * x_step
    y = height / 2 - piecewise_function(x / width)
    points.append((int(x), int(y)))

# 使用采样的坐标点绘制黑色折线
draw.line(points, fill='black', width=2)

# 将PIL图像转换为OpenCV可以处理的格式（numpy数组）
image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# 保存图片，可以根据需要修改保存路径和文件名
cv2.imwrite('result_image.png', image_cv2)