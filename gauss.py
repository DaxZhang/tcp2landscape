from PIL import Image, ImageDraw
import numpy as np
import math

def calculate_linear_function(point1, point2):
    """
    根据两个点的坐标计算对应的一次函数的斜率和截距

    参数:
    point1 (tuple): 第一个点的坐标，格式为 (x1, y1)
    point2 (tuple): 第二个点的坐标，格式为 (x2, y2)

    返回:
    tuple: 包含斜率 (k) 和截距 (b) 的元组
    """
    x1, y1 = point1
    x2, y2 = point2
    # 计算斜率k，根据两点间斜率公式 (y2 - y1) / (x2 - x1)
    if x2 - x1 == 0:
        raise ValueError("两点横坐标相同，无法构成一次函数（直线）")
    k = (y2 - y1) / (x2 - x1)
    # 计算截距b，将其中一个点坐标代入y = kx + b 求解b
    b = y1 - k * x1
    return k, b

def linear_function(k,b,x):
    return int(k*x+b)



def generate_line_segments(num_segments, num_samples):
    """
    生成指定数量的折线段
    """
    width = 800
    height = 450
    left_point = (-200, 0.99*height)  # 起点（在图片左侧外面）
    right_point = (width + 200, 0)  # 终点（在图片右侧外面）
    mid_height = (left_point[1]+right_point[1])//2
    k,b = calculate_linear_function(left_point,right_point)
    print(k,b)
    image = Image.new('RGB', (width, height), 'white')

    draw = ImageDraw.Draw(image)
    # draw.line([left_point,right_point], fill='black', width=2)

    for _ in range(num_segments):
        start_point = left_point
        points = [start_point]
        current_point = [0,0]
        
        for _ in range(num_samples):
            # 从正态分布采样一个方向（这里简单示例，你可以调整参数来符合更合理的分布情况）
            
            # 随机生成一个步长（示例范围，可调整）
            step_length = np.random.randint(200, 400)
            # 根据方向和步长计算下一个点的坐标
            print(step_length)
            new_x = current_point[0] + step_length
            mid_y = linear_function(k,b,(current_point[0] + step_length))
            
            new_y = int(np.random.normal(mid_y,50))

            if new_x > width or len(points) >= num_samples:
                points.append(right_point)
                draw.line(points, fill='black', width=2)
                break

            
            new_point = (int(new_x), int(new_y))
            points.append(new_point)
            current_point = new_point

            # 判断横向距离是否超过最右点或者达到折线段最大点数
            
        print(points)

    return image


if __name__ == "__main__":
    num_segments = 2  # 折线条数
    num_samples = 3  # 每条折线段上的采样点数
    result_image = generate_line_segments(num_segments, num_samples)
    result_image.save('lines_image.png')