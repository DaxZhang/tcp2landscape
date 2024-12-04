from PIL import Image, ImageDraw
import numpy as np
import math

GR = (2.23606797749979 - 1) / 2


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








def triangular_func(x,w=2,phi=0,A=100):
    return A * np.sin(w * x+phi)


def linear_function(k,b,x):
    return int(k*x+b)


def interval_overlap(a,b):
    
    if (a[1] < b[0] or a[0] > b[1]):
        return False
    if (a[1]<b[1] and a[0] > b[0]) or (a[1]>b[1] and a[0] < b[0]):
        return True
    r = [a[0],a[1],b[0],b[1]]
    r.sort()
    print(r)
    if (r[-1]- r[0])/(r[1]-r[2]) > 2:
        return False
    return True



def generate_line_segments(num_segments, num_samples, num_pulse):
    """
    生成指定数量的折线段
    """
    width = 1600
    height = 450
    left_point = (-200, 0.8*height)  # 起点（在图片左侧外面）
    right_point = (width + 200, 0)  # 终点（在图片右侧外面）
    mid_height = (left_point[1]+right_point[1])//2
    k,b = calculate_linear_function(left_point,right_point)
    print(k,b)
    image = Image.new('RGB', (width, height), 'white')

    draw = ImageDraw.Draw(image)
    # draw.line([left_point,right_point], fill='black', width=2)

    pulses = []
    for i in range(num_pulse):

        while True:
            start_point = np.random.uniform(0,width)
            om = np.random.uniform(0, 4)
            w = 1/50/(om+1)
            t =  2 * np.pi / w
            end_point = start_point + t /2

            flag = True
            for p in pulses:
                if interval_overlap(p,[start_point,end_point]):
                    print(p,[start_point,end_point])
                    flag = False
            if flag:
                pulses.append([start_point,end_point])
                break



        mid = (start_point+end_point) / 2
        phi = - w * start_point

        vr = np.random.normal(GR*height,50)
        A_range = vr
        A = np.random.uniform(A_range-100,A_range)

        x1 = np.random.uniform((start_point+mid)/2,mid)
        x2 = np.random.uniform(mid,(end_point+mid)/2)
        y1 = triangular_func(x1,w,phi+np.pi,A) + vr
        y2 = triangular_func(x2,w,phi+np.pi,A) + vr
        
        pts = [
            (int(start_point), height // 2+ vr),
            (int(x1),int(y1)),
            (int(x2),int(y2)),
            (int(end_point), height // 2+ vr)
        ]
        
        draw.line(pts,fill='black',width=2)

        pulse_pts = pts



    # 生成背景线
    background_pts = []
    for i in range(num_segments):
        start_point = left_point
        points = [start_point]
        current_point = [0,0]
        samples = 200
        x_step = width/samples
        x = 0
        pt=[]

        
        om = np.random.uniform(3, 6)
        w = 1/50/(om+1)
        t =  2 * np.pi / w
        round = (int(width / t)+1) * 4
        
        oa = np.random.uniform(1,3)

        phi = w * np.random.uniform(0,t)

        vr = np.random.normal(GR * height, 50)
        

        x_n = -phi
        mid = 0
        pts = [start_point]


        for _ in range(round):
            mid = x_n + t / 4
            # x1 = np.random.normal((x_n+mid)/2,t/16)
            # x2 = np.random.normal((x_n+mid + t/2)/2,t/16)
            x1 = np.random.uniform(x_n, mid)
            x2 = np.random.uniform(mid,x_n+t/2)
            y1 = triangular_func(x1,w,phi=phi, A=50*oa) + vr
            y2 = triangular_func(x2,w, phi=phi, A=50*oa) + vr
            x_n += t / 2
            
            eps = np.random.uniform(0,1)
        
            if eps > 0:
                pts.append((x1,y1))
                pts.append((x2,y2))
                draw.line(pts,fill='black',width=2)
          
            pts = [(x2,y2)]


        # points.append(right_point)
        # draw.line(points,fill='black',width=2)
            

        for m in range(samples):
            x += x_step
            y = triangular_func(x,w,phi=phi,A=50*oa)+vr
            pt.append((int(x),int(y)))
        # draw.line(pt,fill='blue',width=1)
        
        # for j in range(num_samples):
        #     # 从正态分布采样一个方向（这里简单示例，你可以调整参数来符合更合理的分布情况）
            
        #     # 随机生成一个步长（示例范围，可调整）
            
            
        #     step_length = np.random.randint(120,200)
        #     # 根据方向和步长计算下一个点的坐标
        #     print(step_length)
        #     new_x = current_point[0] + step_length
        #     mid_y = linear_function(k,b,(current_point[0] + step_length))
            
        #     new_y = triangular_func(new_x,w=1/50/(i+1),phi=(i)/num_segments*2*np.pi,A=30 * (i+0.5))+height//2

        #     # 判断横向距离是否超过最右点或者达到折线段最大点数

        #     if new_x > width or len(points) >= num_samples:
        #         points.append(right_point)
        #         # draw.line(points, fill='black', width=2)
        #         break

            
        #     new_point = (int(new_x), int(new_y))
        #     points.append(new_point)
        #     current_point = new_point

            
            
        print(points)


    # 生成前景线
    foreground_pts = []
    for i in range(num_samples):
        pass

    return image, pulse_pts, background_pts, foreground_pts


def generate_perpendicular_tcp(num_flow,num_samples):
    width = 450
    height = 800
    up_point = (0,-200)  
    min_distance= 100
    image = Image.new('RGB', (width,height), 'white')
    draw = ImageDraw.Draw(image)

    down_point_xs =[]
    limit = 0
    while len(down_point_xs) < num_flow:
        limit+=1
        dpx = np.random.randint(100, 450)
        if not down_point_xs or all(abs(dpx - existing_y) >= min_distance for existing_y in down_point_xs):
            down_point_xs.append(dpx)
        if limit > 10000:
            down_point_xs=[]
            limit=0
            continue

    for i in range(num_flow):
        down_point = (down_point_xs[i],height+200)
        k, b = calculate_linear_function(up_point,down_point)
        current_point = [0,0]
        points = [up_point]

        for j in range(num_samples):
            step_length = int(np.random.normal((down_point[0]-up_point[0])/num_samples,5))
            print(i, j, step_length,down_point[0])
            new_x = current_point[0] + step_length
            mid_y = linear_function(k,b,(new_x))
            new_y = np.random.normal(mid_y,50)

            if new_y > height or len(points) >= num_samples:
                points.append(down_point)
                draw.line(points, fill='black', width=2)
                break
            new_point = (int(new_x), int(new_y))
            points.append(new_point)
            current_point = new_point

    return image
    

def generate_image():
    # 创建一个空白图像，这里设置图像大小为400x400，背景颜色为白色（RGB模式下 (255, 255, 255)）
    image = Image.new('RGB', (400, 400), (255, 255, 255))
    draw = ImageDraw.Draw(image)

    # 确定左侧起始点的坐标（这里坐标范围是基于图像大小设定的，可按需调整）
    start_x = 50
    start_y = 200

    # 确定右侧三个点共同的横坐标（可根据图像宽度等情况调整合适的值）
    right_x = 300

    # 随机生成三个纵坐标，保证间距不要太小，这里设置最小间距为10，你可以根据需求调整
    min_distance = 10
    y_list = []
    while len(y_list) < 3:
        new_y = np.random.randint(50, 350)
        if not y_list or all(abs(new_y - existing_y) >= min_distance for existing_y in y_list):
            y_list.append(new_y)

    # 绘制起始点（红色）
    draw.ellipse((start_x - 5, start_y - 5, start_x + 5, start_y + 5), fill=(255, 0, 0))

    # 绘制右侧三个点（蓝色）
    for y in y_list:
        draw.ellipse((right_x - 5, y - 5, right_x + 5, y + 5), fill=(0, 0, 255))

    # 连线（绿色）
    for y in y_list:
        draw.line([(start_x, start_y), (right_x, y)], fill=(0, 255, 0), width=2)

    # 保存图像为文件，可指定不同格式（如PNG、JPEG等）及路径，这里保存为当前目录下的 result.png
    image.save('result.png')

if __name__ == "__main__":
    num_segments =  3 # 折线条数
    num_samples = 5  # 每条折线段上的采样点数
    num_pulse = 1
    result_image = generate_line_segments(num_segments,num_samples,num_pulse)
    result_image.save(f'lines_image.png')
    

    # for i in tqdm.tqdm(range(100)):
    #     result_image = generate_perpendicular_tcp(num_segments, num_samples)
    #     result_image.save(f'./perpendicular/lines_image_{i}.png')
        