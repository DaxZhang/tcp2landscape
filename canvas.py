from PIL import Image,ImageDraw
import numpy as np
from stream import *



class Canvas():
    def __init__(self, width = 1600, height = 450):
        self.width = width
        self.height = height

        image = Image.new('RGB', (self.width,self.height), 'white')
        self.image = image
        self.bgs_size = 0

        
        drawer = ImageDraw.Draw(image)
        self.drawer = drawer
        
        self.bgs = []
        self.fgs = []
        self.pls = []
        self.ves = []

        # 偏置
        self.bias_h = 0
        self.bias_v = 0

    def draw(self):
        for lines in self.bgs:
            self.drawer.line(lines, fill='blue', width=4)
        for lines in self.fgs:
            self.drawer.line(lines,fill='gray',width=6)
        for lines in self.pls:
            self.drawer.line(lines,fill='black',width=3)
        for lines in self.ves:
            self.drawer.line(lines,fill='green', width=3)

    def add_background_stream(self):
        # TODO: initialize based on the bias
        # define the triangular function
        omega = 1/(self.width / np.random.uniform(3, 10))
        t = 2 * np.pi / omega
        
        oa = 3
        
        phi = 3*np.pi /2-omega*(0.5 - self.bias_h)*self.width
        phi = np.random.normal(phi, np.pi / 12 *omega)
        if self.bgs_size == 0:
            phi = omega*np.random.uniform(0,t)

        vr = np.random.normal((0.5) * self.height, 50)

        now_x = 0
        now_y = triangular_func(now_x,omega,phi=phi,A = 50 * oa) + vr

        stream = [(int(now_x), int(now_y))]
        
        step = t / 4
        while now_x < self.width:
            now_x += np.random.normal(step,step / 4)
            if now_x >= self.width:
                now_x = self.width
            now_y = triangular_func(now_x,omega,phi=phi,A = 50 * oa) + vr
            
                
            stream.append((int(now_x),int(now_y)))


        self.bgs.append(stream)
        self.bgs_size += 1
        self.update_bias()


    def add_foreground_stream(self):

        omega = 1/(self.width /  np.random.uniform(6, 11))

        t = 2 * np.pi / omega
        
        oa = np.random.uniform(1,2)
        phi = omega * np.random.uniform(0,t)

        vr = np.random.normal(0.9 * self.height, 50)

        now_x = 0
        now_y = triangular_func(now_x,omega,phi=phi,A = 50 * oa) + vr

        stream = [(int(now_x), int(now_y))]
        
        step = t / 4
        while now_x < self.width:
            now_x += np.random.normal(step,step/4)
            now_y = triangular_func(now_x,omega,phi=phi,A = 50 * oa) + vr
            stream.append((int(now_x),int(now_y)))


        self.fgs.append(stream)

        self.update_bias()



    def save_(self, filname = 'result_image.png'):
        self.image.save(f'./{filname}')

    def add_peaks(self,stream_pts = None, num_peaks = 40):
        if len(self.bgs) == 0:
            self.add_background_stream()
        if stream_pts == None:
            stream_pts = self.bgs[0]
            

        left = stream_pts[0][0]
        right = stream_pts[-1][0]
        pulses = []
        for i in range(num_peaks):

            while True:
                # print(max(left,0), min(right,self.width))
                start_point = np.random.uniform(max(left,0), min(right,self.width))
                
                om = np.random.uniform(0, 1)
                w = 1/10/(om+1)
                t =  2 * np.pi / w
                end_point = start_point + t /2

                flag = True
                for p in pulses:
                    if interval_overlap(p,[start_point,end_point]):
                        print(p,[start_point,end_point])
                        flag = True
                        break
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
            

            self.pls.append(pts)

    def add_verts(self, stream_pts, up = True):   
        add_pt, idx, p, a = self.make_breakpoint(stream_pts)
        hat = add_pt
        
        if up:
            hat = (add_pt[0], add_pt[1]-100)
        else:
            hat = (add_pt[0], add_pt[1]+100)

        self.ves.append([add_pt,hat])

        self.update_bias()
        

    def make_breakpoint(self,stream_pts,on_left = True):
        left = max(0,stream_pts[0][0])
        right = min(stream_pts[-1][0],self.width)

        x = 0
        if on_left:
            x = left + (right - left) * GR
        else:
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
        return (int(x), y), ind, pred, after
    
    def update_bias(self):
        # 根据bgs fgs的点计算水平和竖直偏移量

        bias = 0
        for line in self.bgs:
            for idx,point in enumerate(line):

                x = point[0]
                y = point[1]

                if idx == 0:
                    continue

                point_pred = line[idx-1]
                xp = point_pred[0]
                yp = point_pred[1]
                xp_b = 1 if xp > self.width/2 else -1
                yp_w =  (self.height-yp) / self.height
                p_b = xp_b * yp_w
                

                x_bias = 1 if x > self.width/2 else -1 # min((x - self.width/2)/(self.width/2),1)
                y_weight = (self.height-y) / self.height
            
                perp_bias = x_bias*y_weight

                balance = (perp_bias + p_b) / 2

                leng = (x - xp)/self.width
                bias += balance * leng

        for line in self.fgs:
            for idx,point in enumerate(line):

                x = point[0]
                y = point[1]

                if idx == 0:
                    continue

                point_pred = line[idx-1]
                xp = point_pred[0]
                yp = point_pred[1]
                xp_b = 1 if xp > self.width/2 else -1
                yp_w =  (self.height-yp) / self.height
                p_b = xp_b * yp_w
                

                x_bias = 1 if x > self.width/2 else -1 # min((x - self.width/2)/(self.width/2),1)
                y_weight = (self.height-y) / self.height
            
                perp_bias = x_bias*y_weight

                balance = (perp_bias + p_b) / 2

                leng = (x - xp)/self.width
                bias += balance * leng
        
        for vert in self.ves:
            pt1 = vert[0]
            pt2 = vert[1]

            x = pt1[0]
            x_bias = (self.width/2-x)/(self.width)/2
            y_weight = abs(pt1[1]-pt2[1])/self.height
            perp_bias = x_bias*y_weight
            bias += perp_bias



        
        self.bias_h = bias

        


        

    def segment_stream(self):
        # 根据虚实关系打断线
        pass

    def render_(self):
        # 根据折线匹配图像并放进去
        pass

    
        

            

if __name__ == '__main__':
    canvas = Canvas()
    canvas.add_background_stream()
    canvas.add_background_stream()
    # canvas.add_foreground_stream()
    # canvas.add_peaks()
    # canvas.add_verts(canvas.fgs[0])
    canvas.update_bias()
    print(f'平衡度为：{canvas.bias_h}')

    canvas.draw()

    canvas.save_()
    
