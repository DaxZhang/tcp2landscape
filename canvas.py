from PIL import Image,ImageDraw
import numpy as np
from stream import *

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', type=int, default=1600)

    parser.add_argument('--height', type=int, default=450)
    parser.add_argument('--filename', type=str, default='result.png')

    args = parser.parse_args()
    return args




class Canvas():
    def __init__(self, args):
        self.width = args.width
        self.height = args.height

        image = Image.new('RGB', (self.width,self.height), 'white')
        self.image = image
        self.bgs_size = 0

        self.clock = 0
        self.v = 5
        self.t = 0
        self.dense = 0
        self.dense_y = 0
        
        
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
            # self.draw_tree_(lines)
            
            self.drawer.line(lines,fill='green', width=3)

    def clean(self):
        self.image = Image.new('RGB', (self.width,self.height), 'white')
        self.drawer = ImageDraw.Draw(self.image)

    def draw_tree_(self,tree):
        pt1 =np.array(tree[0])

        pt2 = np.array(tree[1])

        scale = np.linalg.norm(pt1-pt2)


        line_left_base = (int(pt1[0]-0.5*scale*(1-0.618)),pt1[1])
       

        line_left_end = (int(line_left_base[0] - scale * np.sin(np.radians(5))), int(line_left_base[1] - (0.618+0.618*(1-0.618))* scale * np.cos(np.radians(5))))

        line_right_base =(int(pt1[0]+0.5*scale*0.618),pt1[1])
        
        line_right_end = (int(line_right_base[0] + scale * np.sin(np.radians(10))), int(line_right_base[1] -0.618*scale * np.cos(np.radians(10))))


        self.drawer.line(tree,fill='green',width=3)
        self.drawer.line([line_left_base,line_left_end],fill='green',width=3)
        self.drawer.line([line_right_base,line_right_end],fill='green',width=3)


        

        



    def add_background_stream(self):
        # TODO: initialize based on the bias
        # define the triangular function
        omega = 1/(self.width / np.random.uniform(3,5))
        
        
        t = 2 * np.pi / omega
        
        a = np.random.uniform(100,150)

        
        phi = 3*np.pi /2-omega*self.width
        phi = np.random.normal(phi, np.pi / 4 *omega)
        if self.bgs_size == 0:
            phi = omega*np.random.uniform(0,t)

        



        self.clock += 1
        vr = np.random.normal(self.clock * 0.25 * self.height, 5)
        if self.clock > 1:
            if self.bias_h <= 0:
                pt_last = [0, self.height]
                if self.clock == 2:
                    pt_last = [0]
                for s in self.bgs:
                    if s[-1][1] < self.height and s[-1][1] > 0:
                        pt_last.append(s[-1][1])
                if len(pt_last) == 1:
                    pt_last.append(self.height)
                pt_last.sort()
                dis = np.diff(pt_last)
                idx = np.argmax(dis)

                y1 = pt_last[idx]
                y2 = pt_last[idx+1]
                 
                pt_last_y = y1 + 0.618*(y2 - y1)
                omega = np.random.uniform(1.25*np.pi/self.width, 2 * np.pi/self.width)
                

                phi = np.random.uniform(1.5*np.pi,1.75*np.pi) - omega * self.width

                vr = self.height/2 - self.dense_y*self.height/2

                a = (pt_last_y - vr)/triangular_func(self.width,omega,phi,1)


                if abs(self.dense_y) < 2e-1:
                    omega = 1 / (self.width/np.random.uniform(3,7.5))
                    phi = np.random.uniform(1.5*np.pi,1.9*np.pi) - omega * self.width
                    vr = pt_last_y + 0.618 * (self.height - pt_last_y)
                    a = (pt_last_y - vr)/triangular_func(self.width,omega,phi,1)
                    if a < 0:
                        a = -a
                        phi += np.pi


                t = 2 * np.pi / omega


                print(pt_last)
                print(dis)
                print(f'y1 = {y1}, y2 = {y2}, y = {pt_last_y}')
                

                
                # if abs(self.dense_y) > 1e-1:
                #     vr =self.height/2 - self.dense_y*self.height/2
                #     a = abs(pt_last_y - vr)
                # else:
                #     vr = pt_last_y + a

                # phi = np.pi
                # if vr > pt_last_y:
                #     phi += 0.5 * np.pi

                # else:
                #     phi -= 0.5 * np.pi
                    


                

                # phi = phi-omega*self.width
                # # phi = np.random.uniform(phi, np.pi / 4 *omega)
                print(np.degrees(phi), omega, a, vr, triangular_func(self.width,omega,phi,a) + vr)
                
            else:
                pt_last = [0, self.height]
                if self.clock == 2:
                    pt_last = [0]
                for s in self.bgs:
                    if s[0][1] < self.height and s[0][1] > 0:
                        pt_last.append(s[0][1])
                if len(pt_last) == 1:
                    pt_last.append(self.height)
                pt_last.sort()
                dis = np.diff(pt_last)
                idx = np.argmax(dis)

                y1 = pt_last[idx]
                y2 = pt_last[idx+1]
                 
                pt_last_y = y1 + 0.618* (y2 - y1)
                omega = np.random.uniform(1.25*np.pi/self.width, 2* np.pi/self.width)

                phi = np.random.uniform(1.25*np.pi, 1.5*np.pi)

                vr = self.height/2 - self.dense_y*self.height/2

                a = (pt_last_y - vr)/triangular_func(0,omega,phi,1)
                if abs(self.dense_y) < 2e-1:
                    omega = 1 / (self.width/np.random.uniform(3,7.5))
                    vr = pt_last_y + 0.618 * (self.height - pt_last_y)
                    phi = np.random.uniform(1.1*np.pi, 1.5*np.pi)
                    a = (pt_last_y - vr)/triangular_func(0,omega,phi,1)
                    if a < 0:
                        a = -a
                        phi += np.pi
                print(pt_last)
                print(dis)
                print(f'y1 = {y1}, y2 = {y2}, y = {pt_last_y}')
                t = 2 * np.pi / omega
                
                # if abs(self.dense_y) > 1e-1:
                #     vr =self.height/2 - self.dense_y*self.height/2
                #     a = abs(pt_last_y - vr)
                # else:
                #     vr = pt_last_y + a

                # phi = np.pi
                # if vr > pt_last_y:
                #     phi += 0.5 * np.pi

                # else:
                #     phi -= 0.5 * np.pi

                # omega = np.random.uniform(1.25*np.pi/self.width, 1.5* np.pi/self.width)
                # phi = phi
                # # phi = np.random.normal(phi, np.pi / 4 *omega)
                print(np.degrees(phi), omega, a, vr, triangular_func(0,omega,phi,a) + vr)
                

        now_x = 0
        now_y = triangular_func(now_x,omega,phi=phi,A = a) + vr

        stream = [(int(now_x), int(now_y))]
        
        step = t / 7
        while now_x < self.width:
            now_x += np.random.normal(step,step / 7)
            if now_x >= self.width:
                now_x = self.width
            now_y = triangular_func(now_x,omega,phi=phi,A = a) + vr
            
                
            stream.append((int(now_x),int(now_y)))


        self.bgs.append(stream)
        self.bgs_size += 1
        self.update_bias()
    

    def add_foreground_stream(self):


        self.clock+=1

        omega = 1/(self.width /  np.random.uniform(6, 11))

        t = 2 * np.pi / omega
        
        oa = np.random.uniform(20,self.height/20)
        phi = omega * np.random.uniform(0,t)

        vr = np.random.normal(self.clock * 0.1666 * self.height, 10)

        now_x = 0
        now_y = triangular_func(now_x,omega,phi=phi,A = oa) + vr

        stream = [(int(now_x), int(now_y))]
        
        step = t / 4
        while now_x < self.width:
            now_x += np.random.normal(step,step/4)
            now_y = triangular_func(now_x,omega,phi=phi,A = oa) + vr
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

        if self.bias_h > 0:
            add_pt, idx, p, a = self.make_breakpoint(stream_pts,False)
        else:
            add_pt, idx, p, a = self.make_breakpoint(stream_pts,True)
        hat = add_pt
        add_pt = (add_pt[0], add_pt[1]+25)
        h = 0.618 * self.height
        if up:
            hat = (add_pt[0], add_pt[1]-h)
        else:
            hat = (add_pt[0], add_pt[1]+h)

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
        vert = []
        for line in self.bgs:
            for idx,point in enumerate(line):
                vert.append(point[1])
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
                vert.append(y)
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

        self.vert = np.mean(vert)
        self.bias_h = 0
        self.bias_v = (self.vert-self.height/2)/(self.height/2)


        self.bias_h += self.density()


        y = []
        w = 0
        for stream in self.bgs:
            for i, pt in enumerate(stream):
                if i == 0:
                    continue
                pt_pred = stream[i-1]
                x1,y1 = pt[0],pt[1]
                x2,y2 = pt_pred[0],pt_pred[1]
                if y1 < 0:
                    y1 = 0
                if y2 < 0:
                    y2 =0
                if y1 > self.height:
                    y1 = self.height
                if y2 > self.height:
                    y2 = self.height
                
                y.append((y1+y2)/2 )
                
                
        
        self.dense_y = (np.mean(y)-self.height/2)/(self.height/2) 




    
    def density_at(self, x):
        y = []
        for stream in self.bgs:
            y.append(find_point(stream,x))
        
        if len(y) == 1:
            y.append(0)
        y.sort()
        # if y[0] >= 0:
        #     y.insert(0,0)
        
        y = np.array(y)

        dis = np.diff(y)
        var = np.var(dis)
        avg = np.mean(dis)

        return - avg/self.height + var/self.height**2


    
    def density(self):
        num = 10
        stp = np.linspace(0, self.width,num)

        l_dense = []
        left_stp = stp[:num//2]
        for x in left_stp:
            l_dense.append(self.density_at(x))
        left_dense = np.average(l_dense)

        r_dense = []
        right_stp = stp[num//2:]
        for x in right_stp:
            r_dense.append(self.density_at(x))
        right_dense = np.average(r_dense)
        self.dense = right_dense - left_dense
        return self.dense

        

    def segment(self, stream):
        segments = []
        now_seg = []
        is_pred_visi = False
        for i,pt in enumerate(stream):
            # print("segs: ",segments)
            # print("now segs: ",now_seg)
            # print(self.is_visible(pt),'before',is_pred_visi)

            if self.is_visible(pt):
                if is_pred_visi == False:
                    is_pred_visi = True
                    if i != 0:
                        pred_pt = stream[i-1]
                        k,b = calculate_linear_function(pred_pt,pt)
                        if pred_pt[0] < 0:
                            x = 0
                            y = int(b)
                        if pred_pt[1] < 0:
                            y = 0
                            x = linear_function(1/k,-b/k,y)
                        if pred_pt[1] > self.height:
                            y = self.height
                            x = linear_function(1/k,-b/k,y)
                        add_pt = (int(x),int(y))
                        now_seg.append(add_pt)
                        now_seg.append(pt)
                            
                    else:
                        now_seg.append(pt)
                else:
                    now_seg.append(pt)
                if i == len(stream) - 1:
                    segments.append(now_seg.copy())
                    now_seg.clear()
                
            else:
                if is_pred_visi == False:
                    continue
                else:
                    is_pred_visi = False
                    pred_pt = stream[i-1]
                    k,b = calculate_linear_function(pred_pt,pt)
                    if pt[0] > self.width:
                        x = self.width
                        y = linear_function(k,b,x)
                    if pt[1] < 0:
                        y = 0
                        x = linear_function(1/k,-b/k,y)
                    if pt[1] > self.height:
                        y = self.height
                        x = linear_function(1/k,-b/k,y)
                    add_pt = (int(x),int(y))
                    now_seg.append(add_pt)
                    segments.append(now_seg.copy())
                    now_seg.clear()
        return segments

    def is_visible(self,pt):
        if pt[0] < 0 or pt[0] > self.width:
            return False
        
        if pt[1] < 0 or pt[1] > self.height:
            return False
        
        return True
    
    def add_tree(self):
        best_deviation = np.inf
        best_pt = None
        best_pred = None
        best_succ= None
        for segment in self.bgs:
            
            add_pt, _, pred, succ =make_breakpoint(segment)
            x_dva = min(
                (add_pt[0] - self.width/3)**2,
                (add_pt[0] - 2* self.width/3) **2,
            )/self.width**2
            y_dva = (add_pt[1] - 2 * self.height/3)**2/self.height**2
            # lor = 1 if add_pt[0] > self.width else -1
            dva = x_dva + y_dva 
            if dva < best_deviation:
                best_deviation = dva
                best_pt = add_pt
                best_pred = pred
                best_succ = succ
            
            add_pt, _, pred,succ =make_breakpoint_right(segment)
            x_dva = min(
                (add_pt[0] - self.width/3)**2,
                (add_pt[0] - 2* self.width/3) **2,
            )/self.width**2
            y_dva = (add_pt[1] - 2 * self.height/3)**2/self.height**2
            dva = x_dva + y_dva
            if dva < best_deviation:
                best_deviation = dva
                best_pt = add_pt
                best_succ = succ
                best_pred = pred
                
        k,b = calculate_linear_function(best_pred, best_succ)
        print(f'best dva: {dva}')
        # print(f'best pred: {best_pred}, best succ: {best_succ}')
        self.drawer.circle(best_pred,4,fill='black')
        self.drawer.circle(best_succ,4,fill='black')
        # print(k,b)

        dgr = np.arctan(-k)
        # print(np.degrees(dgr))
        basept = (best_pt[0] ,best_pt[1]+25)
            
        endpt = (int(basept[0] - 100* np.sin(dgr)), int(basept[1] - 100*np.cos(dgr)))

        self.ves.append([basept, endpt])
        print(self.ves)



        


        

            

    def segment_stream(self):
        temp = []
        
        for stream in self.bgs:
            print("before segment: ",stream)

            segs = self.segment(stream)

            print('after segment: ',segs)
            temp+=segs

        self.bgs = temp



    def render_(self):
        # 根据折线匹配图像并放进去
        pass

    
        

            

if __name__ == '__main__':

    args = parse_arguments()
    canvas = Canvas(args)
    canvas.add_background_stream()

    canvas.draw()
    canvas.save_('result1.png')


    print(f'平衡度为：{canvas.bias_h}, {canvas.dense_y}')
    print(f"dense= {canvas.dense}")
    canvas.add_background_stream()
    canvas.draw()
    canvas.save_('result2.png')
    print(f'平衡度为：{canvas.bias_h}, {canvas.dense_y}')
    print(f"dense= {canvas.dense}")
    canvas.add_background_stream()
    print(f'平衡度为：{canvas.bias_h}, {canvas.dense_y}')
    print(f"dense= {canvas.dense}")
    canvas.draw()

    canvas.save_('result_bef.png')

    canvas.clean()
    canvas.segment_stream()
    canvas.add_tree()

    # canvas.add_verts(canvas.bgs[-1])
    # for _ in range(5):
    #     canvas.add_foreground_stream()
    # canvas.add_peaks()
    # canvas.update_bias()


    canvas.draw()

    canvas.save_(args.filename)
    
