from PIL import Image, ImageDraw
import numpy as np
import math
from scipy.spatial.distance import euclidean

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
        print(x1)
        raise ValueError("两点横坐标相同，无法构成一次函数（直线）")
    k = (y2 - y1) / (x2 - x1)
    # 计算截距b，将其中一个点坐标代入y = kx + b 求解b
    b = y1 - k * x1
    return k, b





def triangular_func(x,w=2,phi=0,A=100):
    return A * np.sin(w * x+phi)


def linear_function(k,b,x):
    return int(k*x+b)


K = 3.5

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


def find_point(stream, x):
    ind = -1
    for pt in stream:
        # if abs(pt[0] - x) < 1e-5:
        if pt[0] == x:
            return pt[1]
        elif pt[0] < x:
            ind += 1
        else:
            # print("pt[0]:", pt[0], "        x:", x,"           ind:", ind)
            break

    if ind== -1 or ind == len(stream)-1:
        return None
    # x is between pred and succ
    # print("pred_ind", ind)
    pred = stream[ind]
    succ = stream[ind+1]

    k,b = calculate_linear_function(pred,succ)

    y = linear_function(k,b,x)
    return int(y)





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
    for i in range(0):

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
                    flag = True
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


def draw(w, h, b_pts,f_pts, p_pts,v_pts):
    image = Image.new('RGB', (w,h), 'white')
    drawer = ImageDraw.Draw(image)
    for lines in b_pts:
        drawer.line(lines, fill='blue', width=4)
    for lines in f_pts:
        drawer.line(lines,fill='gray',width=6)
    for lines in p_pts:
        drawer.line(lines,fill='black',width=3)
    for lines in v_pts:
        drawer.line(lines,fill='green', width=3)
    
    return image


def generate_peak(stream_pts, num_peaks, width = 1600, h = 450):

    left = stream_pts[0][0]
    right = stream_pts[-1][0]
    pulses = []
    pulse_pts = []
    for i in range(num_peaks):

        while True:
            print(max(left,0), min(right,width))
            start_point = np.random.uniform(max(left,0), min(right,width))
            
            om = np.random.uniform(0, 1)
            w = 1/10/(om+1)
            t =  2 * np.pi / w
            end_point = start_point + t /2

            flag = True
            for p in pulses:
                if interval_overlap(p,[start_point,end_point]):
                    print(p,[start_point,end_point])
                    flag = True
            if flag:
                pulses.append([start_point,end_point])
                break
            break
            



        mid = (start_point+end_point) / 2
        phi = - w * start_point


        x = mid

        pred = stream_pts[0]
        ind = 0
        after = stream_pts[-1]
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

        height = y


        vr = np.random.normal(height,10)
        A_range = 60
        A = np.random.uniform(10,A_range)

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
        

        pulse_pts.append(pts)
    
    return pulse_pts



def segment_stream(stream_pts):
    left = stream_pts[0][0]
    right = stream_pts[-1][0]
    print(left,right)

    mode = np.random.randint(0,4)
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

        left_pts:list = stream_pts[:ind]
        right_pts:list = stream_pts[ind:]

        # 加点，分割完毕
        left_pts.append(add_pt)
        right_pts.insert(0,add_pt)

        left_gr_p, l_id, p, a = make_breakpoint(left_pts)

        left_pts = left_pts[:l_id]
        left_pts.append(left_gr_p)
        
        
        return [left_pts, right_pts]

    elif mode == 1:
        # right Golden Section
        add_pt, idx, p, a = make_breakpoint(stream_pts)

        left_pts:list = stream_pts[:idx]
        right_pts:list = stream_pts[idx+1:]
        
        return left_pts, right_pts
    
    elif mode == 2:
        # identify
        add_pt, idx, p, a = make_breakpoint_right(stream_pts)

        left_pts:list = stream_pts[:idx-1]
        right_pts:list = stream_pts[idx:]
        
        return left_pts, right_pts
        
    elif mode == 3:

        return [stream_pts]
    elif mode == 4:
        return stream_pts
    elif mode == 5:
        return stream_pts
    elif mode == 6:
        return stream_pts
    
def closest_point_on_line_segment(p1, p2, point):
    """
    计算给定点到线段的最近点。
    :param p1: 线段的第一个端点 (x1, y1)
    :param p2: 线段的第二个端点 (x2, y2)
    :param point: 给定点 (x, y)
    :return: 线段上离给定点最近的点
    """
    if p1.all() == p2.all():
        return p1
    line_vec = p2 - p1
    point_vec = point - p1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = line_vec / line_len
    point_vec_scaled = point_vec / line_len
    t = np.dot(line_unitvec, point_vec_scaled)
    if t < 0.0:#大于90度，垂足不在线段上
        t = 0.0#落到p1
    elif t > 1.0:#长度超出p2
        t = 1.0#落到p2
    nearest = p1 + t * line_vec
    return nearest

def find_closest_point_on_polylines(polylines, point):
    """
    在一系列折线(polyline)中找到离给定点最近的点。
    :param polyline: 折线，由一系列二维点组成，形状为 (n, 2)
    :param point: 给定点，形状为 (2,)
    :return: 折线中离给定点最近的点
    """
    point_record = []
    dis_record=[]
    for polyline in polylines:
        closest_point = None
        min_distance = float('inf')
        for i in range(len(polyline) - 1):
            p1 = np.array(polyline[i])
            p2 = np.array(polyline[i + 1])
            current_closest = closest_point_on_line_segment(p1, p2, point)
            current_distance = euclidean(current_closest, point)
            
            if current_distance < min_distance:
                closest_point = current_closest
                min_distance = current_distance
        point_record.append(closest_point)
        dis_record.append(min_distance)
    closest_point = point_record[np.argmin(dis_record)]
    closest_point = (int(closest_point[0]), int(closest_point[1]))
    return closest_point

def make_breakpoint(stream_pts,streams):
    """返回stream_pts中依照横坐标进行黄金分割的点"""
    left = max(0,stream_pts[0][0])
    right = stream_pts[-1][0]
    x = left + (right - left) * GR#横向的黄金分割坐标
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
    # print("x, pred, after:",x, pred, after)
    # print(stream_pts)
    k,b = calculate_linear_function(pred,after)
    y = linear_function(k,b,x)
    near_point = find_closest_point_on_polylines(streams, np.array([x,y]))
    return near_point, ind, pred, after


def make_breakpoint_right(stream_pts,streams):
    left = max(0,stream_pts[0][0])
    right = stream_pts[-1][0]
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
    near_point = find_closest_point_on_polylines(streams, np.array([x,y]))
    return near_point, ind, pred, after


def generate_vert_line(stream_pts,up = True):
    add_pt, idx, p, a = make_breakpoint(stream_pts)
    hat = add_pt
    if up:
        hat = (add_pt[0], add_pt[1]-100)
    else:
        hat = (add_pt[0], add_pt[1]+100)

    return [add_pt, hat]




if __name__ == "__main__":
    num_segments =  2 # 折线条数
    num_samples = 5  # 每条折线段上的采样点数
    num_pulse = 1
    result_image,p, b, f = generate_line_segments(num_segments,num_samples,num_pulse)
    result_image.save(f'lines_image.png')
    

    # for i in tqdm.tqdm(range(100)):
    #     result_image = generate_perpendicular_tcp(num_segments, num_samples)
    #     result_image.save(f'./perpendicular/lines_image_{i}.png')
        