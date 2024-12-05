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



def generate_line_segments(num_background, num_pulse, num_foreground):
    """
    生成指定数量的折线段
    """
    width = 1600
    height = 450
    left_point = (-200, int(0.8*height))  # 起点（在图片左侧外面）
    right_point = [width + 200, 0]  # 终点（在图片右侧外面）
    mid_height = (left_point[1]+right_point[1])//2
    k,b = calculate_linear_function(left_point,right_point)
    print(k,b)
    image = Image.new('RGB', (width, height), 'white')

    draw = ImageDraw.Draw(image)
    # draw.line([left_point,right_point], fill='black', width=2)

    pulse_pts = []
    pulses = []
    for i in range(num_pulse):

        while True:
            start_point = np.random.uniform(0,width)
            om = np.random.uniform(0, 1)
            w = 1/10/(om+1)
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

        vr = np.random.normal(GR*(1-GR)*height,100)
        A_range = vr
        A = np.random.uniform(0,A_range)

        x1 = np.random.uniform((start_point+mid)/2,mid)
        x2 = np.random.uniform(mid,(end_point+mid)/2)
        y1 = triangular_func(x1,w,phi+np.pi,A) + vr
        y2 = triangular_func(x2,w,phi+np.pi,A) + vr
        
        pts = [
            (int(start_point), vr),
            (int(x1),int(y1)),
            (int(x2),int(y2)),
            (int(end_point),  vr)
        ]
        
        draw.line(pts,fill='black',width=2)

        pulse_pts.append(pts)



    # 生成背景线
    background_pts = []
    for i in range(num_background):
        start_point = left_point
       
        samples = 200
        x_step = width/samples
        x = 0
        pt=[]

        
        om = np.random.uniform(1, 4)
        w = 1/50/(om+1)
        t =  2 * np.pi / w
        round = (int(width / t)+1) * 4
        
        oa = np.random.uniform(1,2)
        phi = w * np.random.uniform(0,t)

        vr = np.random.normal((1-GR) * height, 50)
        

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
        
            pts.append((int(x1),int(y1)))
            if x1 > width:
                break
            
            pts.append((int(x2),int(y2)))
            if x2 > width:
                break
        print(pts)
        draw.line(tuple(pts),fill='blue',width=2)  
        background_pts.append(pts)

    # 生成前景线
    foreground_pts = []
    for i in range(num_foreground):
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
        
        oa = np.random.uniform(1,2)

        phi = w * np.random.uniform(0,t)

        vr = np.random.normal( 0.9 * height, 30)
        

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
        
            pts.append((int(x1),int(y1)))
            if x1 > width:
                break
            pts.append((int(x2),int(y2)))
            if x2 > width:
                break
          
        draw.line(pts,fill='green',width=2)
   
        foreground_pts.append(pts)


    return image, pulse_pts, background_pts, foreground_pts

def draw(w, h, b_pts,f_pts, p_pts):
    image = Image.new('RGB', (w,h), 'white')
    drawer = ImageDraw.Draw(image)
    for lines in b_pts:
        drawer.line(lines, fill='blue', width=3)
    for lines in f_pts:
        drawer.line(lines,fill='green',width=5)
    for lines in p_pts:
        drawer.line(lines,fill='black',width=2)
    
    return image


def segment_stream(stream_pts):
    left = stream_pts[0][0]
    right = stream_pts[-1][0]
    print(left,right)

    mode = np.random.randint(0,3)
    if mode == 0:
        # left Golden Section
        x = left + (right - left) * GR
        pred = []
        after = []
        ind = 0
        for pt in stream_pts:
            if pt[0] < x:
                pred = pt
                ind += 1
                continue
            if pt[0] > x:
                after = pt
                break

        k,b = calculate_linear_function(pred,after)
        y = linear_function(k,b,x)
        add_pt = (int(x),y)

        stream_pts = stream_pts[:ind]
        stream_pts.append(add_pt)
        
        return stream_pts

    elif mode == 1:
        # right Golden Section
        x = left + (right - left) * (1 - GR)
        pred = []
        after = []
        ind = 0
        for pt in stream_pts:
            if pt[0] < x:
                pred = pt
                ind += 1
                continue
            if pt[0] > x:
                after = pt
                break

        k,b = calculate_linear_function(pred,after)
        y = linear_function(k,b,x)
        add_pt = (int(x),y)

        stream_pts = stream_pts[ind+1:]
        stream_pts.reverse()
        stream_pts.append(add_pt)
        stream_pts.reverse()

        return stream_pts
    
    elif mode == 2:
        # identify
        return stream_pts
    elif mode == 3:

        return stream_pts
    elif mode == 4:
        return stream_pts
    elif mode == 5:
        return stream_pts
    elif mode == 6:
        return stream_pts
    


def make_breakpoint(stream_pts):
    left = stream_pts[0][0]
    right = stream_pts[-1][0]
    x = left + (right - left) * GR
    pred = []
    after = []
    ind = 0
    for pt in stream_pts:
        if pt[0] < x:
            pred = pt
            ind += 1
            continue
        if pt[0] > x:
            after = pt
            break

    k,b = calculate_linear_function(pred,after)
    y = linear_function(k,b,x)
    return (int(x), y)




if __name__ == "__main__":
    num_segments =  2 # 折线条数
    num_samples = 5  # 每条折线段上的采样点数
    num_pulse = 1
    result_image,p, b, f = generate_line_segments(num_segments,num_samples,num_pulse)
    result_image.save(f'lines_image.png')
    

    # for i in tqdm.tqdm(range(100)):
    #     result_image = generate_perpendicular_tcp(num_segments, num_samples)
    #     result_image.save(f'./perpendicular/lines_image_{i}.png')
        