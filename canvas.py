from PIL import Image,ImageDraw
import numpy as np
from stream import *
from contour import get_contour
import os 
import argparse
import torch
import matplotlib.pyplot as plt
import cv2
import time
import math
from geomloss import SamplesLoss
from shapely.geometry import Polygon,Point
from shapely.affinity import translate, scale


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

        self.background_color = (255,255,255)
        self.stream_color = (0,0,255)
        self.tree_color = (0,255,0)
        image = Image.new('RGB', (self.width,self.height), self.background_color)
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
            self.drawer.line(lines, fill=self.stream_color, width=4)
        for lines in self.fgs:
            self.drawer.line(lines,fill='gray',width=6)
        for lines in self.pls:
            self.drawer.line(lines,fill='black',width=3)
        for lines in self.ves:
            # self.draw_tree_(lines)
            
            self.drawer.line(lines,fill=self.tree_color, width=3)

    def clean(self):
        self.image = Image.new('RGB', (self.width,self.height), self.background_color)
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
        
        step = t / 7#t is the period of the triangular function
        while now_x < self.width:
            now_x += np.random.normal(step,step / 7)#plus at list 1/7 of the period
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
        #纠正向右出界，取交点作为最后一个点    
        stream[-1] = (self.width,int(stream[-2][1]+(stream[-1][1]-stream[-2][1])*(self.width-stream[-2][0])/(stream[-1][0]-stream[-2][0])))
        
        if max((row[1]) for row in stream) > self.height:#向下出界
            pred_valid = stream[0][1]<self.height#认为首个点前还有一个相同的点
            current_start_index = 0
            for i in range(len(stream)):
                if pred_valid:
                    if stream[i][1] > self.height:
                        modified_x = int(stream[i-1][0]+(stream[i][0]-stream[i-1][0])*(self.height-stream[i-1][1])/(stream[i][1]-stream[i-1][1]))
                        tmp_stream_pt=(modified_x,self.height)#不能修改原列表，出界的点会在后续用到
                        # tmp_stream_list=stream[current_start_index:i]
                        # tmp_stream_list.append(tmp_stream_pt)
                        self.fgs.append(stream[current_start_index:i]+[tmp_stream_pt])
                        print(self.fgs[-1])
                        pred_valid = False
                    else:
                        pass
                else:#上一个点出界
                    if stream[i][1] > self.height:#当前点也出界
                        pass#处于废掉的一段
                    else:#当前点正常
                        modified_x = int(stream[i-1][0]+(stream[i][0]-stream[i-1][0])*(self.height-stream[i-1][1])/(stream[i][1]-stream[i-1][1]))
                        tmp_stream_pt=(modified_x,self.height)
                        stream[i-1] = tmp_stream_pt#直接将上一个点（出界）吸附到和边界的交点
                        pred_valid = True
                        current_start_index = i-1#该段的起点
            if pred_valid:                        
                self.fgs.append(stream[current_start_index:])
                print(self.fgs[-1])
        else:#未发生向下出界
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
                if (start_point+end_point) / 2 >= stream_pts[-1][0]:
                    print((start_point+end_point) / 2 ,">=",stream_pts[-1][0])
                    continue
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
            
            print("x: ",x,"   stream_pts: ",stream_pts,"    pred: ",pred,"    after: ",after)
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
        """ 返回stream上的黄金分割点"""
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
            # print("stream: ",stream)
            y.append(find_point(stream,x))
        
        if len(y) == 1:
            y.append(0)
        #rid of None
        y = [point for point in y if point is not None]
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
            lor = 1 if add_pt[0] > self.width / 2 else -1
            dva = x_dva + y_dva + self.dense * lor
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
            lor = 1 if add_pt[0] > self.width / 2 else -1
            dva = x_dva + y_dva + self.dense * lor
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


    def point_normalize(self, stream):
        stream = np.array(stream, dtype= np.float32)
        stream = stream.T
        # print(stream)
        stream[0] = stream[0] / (self.width+0.)
        stream[1] = stream[1] / (self.height+0.)

        return stream.T
    
    def point_recover(self,stream):
        stream = np.array(stream,dtype=np.float32)
        stream = stream.T
        stream[0] = stream[0] * self.width
        stream[1] = stream[1] * self.height
        stream = stream.T
        stream = np.array(stream,dtype=np.int32)
        return stream
        


    def optim_stream_and_ply(self,stream, ply):
        ply = np.array(ply)/2
        stream = self.point_normalize(stream)
        
        ply = self.point_normalize(ply)
        epc = 200

        translate = torch.nn.Parameter(torch.tensor([0.,0.],requires_grad=True))
        scale = torch.nn.Parameter(torch.tensor([2.,2.],requires_grad=True))
        ply = torch.from_numpy(ply)
        stream = torch.from_numpy(stream).unsqueeze(1)
        loss_func = SamplesLoss('sinkhorn',blur=0.01)
        loss_record = []

        optimizer = torch.optim.Adam([translate,scale], lr = 0.001)
        for i in range(epc):
            optimizer.zero_grad()

            ply_mod = ply*scale
            ply_mod += translate

            
            ply_mod = torch.clamp_(ply_mod,0.,1.)

            ply_mod_s = ply_mod.unsqueeze(1)
            dist_mat = ply_mod_s-stream.transpose(0,1)
            dist_mat = torch.sum(dist_mat**2, dim=-1)


            min_dist_val,min_dist_ind = torch.min(dist_mat,dim=-1)
            # print(min_dist_ind)
            match_point = stream[min_dist_ind]
            # print(match_point)
            # exit()
            dist = torch.sum((ply_mod - match_point)**2)
            # [g] = torch.autograd.grad(dist,[ply_mod])

            
            dist.backward(retain_graph=True)
            # print(translate.grad)
            # print(dist.grad)
            optimizer.step()
           

            loss_record.append(dist.detach().numpy())

            print(f"epoch {i+1}, loss {dist}.")
        
        print("t: ",translate)
        print("s: ",scale)
        
        e = list(range(epc))
        plt.plot(e,loss_record)
        plt.savefig(f'loss')
        
        ply_mod = ply*scale
        ply_mod += translate
        ply_mod = torch.clamp_(ply_mod,0.,1.)

        print(canvas.point_recover(stream.squeeze(1)))
        print(canvas.point_recover(ply_mod.detach().numpy()))
        
        s = []
        p = []

        for pt in canvas.point_recover(stream.squeeze(1)):
            s.append(tuple(pt))
        for pt in canvas.point_recover(ply_mod.detach().numpy()):
            p.append(tuple(pt))
        

        print(s,p)
        self.drawer.line(s,fill= 'blue',width=2)
        self.drawer.line(p,fill= 'green',width=2)
        canvas.save_()
        
    def plot_polygon(self,stream):
        """plot the stream as a polygon on the canvas
            saved as a np.ndarray with shape (height, width)
            """
        point_list=stream
        #the start point on wich edge
        if(point_list[0][1]<self.height):#start on the left
            point_list.insert(0,(0,self.height))

        if(point_list[-1][1]<self.height):#end on the right
            point_list.append((stream[-1][0],self.height))
        
        point_list.append(point_list[0])#connect the start and end point
        point_list =  np.array([point_list], dtype=np.int32) 
        mask = np.zeros((self.height, self.width), dtype=np.uint8)
        cv2.fillPoly(mask, [point_list], color=255)  # 填充多边形为白色
        # print(mask.shape)(self.height, self.width)
        # mask_single_channel = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # print(mask_single_channel.shape)
        return mask,point_list

    def overlap(self,stream_mask, mat_mask, scale_factor=1,tx=0,ty=0,draw=False):
        """
        stream_mask: the mask of the stream, np.ndarray, shape (height, width), color=0 or 255
        """
        # print(type(mat_mask))
        # print(mat_mask.shape)    
        mat_resized = cv2.resize(mat_mask, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        M = np.float32([[1, 0, tx], [0, 1, ty]])  # 平移矩阵
        mat_transformed = cv2.warpAffine(mat_resized, M, (stream_mask.shape[1], stream_mask.shape[0]))
        overlap_range = cv2.bitwise_and(stream_mask, mat_transformed)
        overlap_area = np.count_nonzero(overlap_range)
        mat_area = np.count_nonzero(mat_transformed)
        stream_area = np.count_nonzero(stream_mask)
        if draw:
            self.draw_overlap(stream_mask, mat_transformed)
        # print("overlap area: ", overlap_area)
        # cv2.imshow("Image 1", stream_mask)
        # cv2.imshow("Image 2", mat_transformed)
        # cv2.imshow("Overlap", overlap_range)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return mat_area+stream_area-overlap_area*2

    def overlap_polygon(self,polygon_1,polygon_2,draw=False):
        canva_polygon=Polygon([(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)])
        polygon_1 = polygon_1.intersection(canva_polygon)
        polygon_2 = polygon_2.intersection(canva_polygon)
        intersect_polygon = polygon_1.intersection(polygon_2)
        intersect_area = intersect_polygon.area
        p1_area = polygon_1.area
        p2_area = polygon_2.area
        if draw:
            img1=self.polygon_to_nparray(polygon_1)
            img2=self.polygon_to_nparray(polygon_2)
            self.draw_overlap(img1,img2)
        return intersect_area+p1_area+p2_area-intersect_area*2.5

    def polygon_to_nparray(self,polygon):
        from matplotlib.path import Path
        # 创建空白图像
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # 创建网格
        x = np.linspace(0, self.width, self.width)
        y = np.linspace(0, self.height, self.height)
        xv, yv = np.meshgrid(x, y)
        points = np.vstack((xv.flatten(), yv.flatten())).T

        # 栅格化多边形
        polygon_path = Path(np.array(polygon.exterior.coords))
        inside = polygon_path.contains_points(points).reshape((self.height, self.width))

        # 填充多边形区域
        mask[inside] = 255

        return mask

    def translate_scale_polygon(self,polygon,tx,ty,scale_factor):
        from matplotlib.path import Path
        Polygon_canvas=Polygon([(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)])
        translated_polygon = translate(polygon, xoff=tx, yoff=ty)
        scaled_polygon = scale(translated_polygon, xfact=scale_factor, yfact=scale_factor, origin=translated_polygon.centroid)
        # scaled_polygon = scaled_polygon.intersection(Polygon_canvas)
        return scaled_polygon
    
    def optim_mask(self,stream,mat_path):#已弃用
        """gradient descent to optimize the mask of the stream"""
        start_time = time.time()
        stream_mask,_ = self.plot_polygon(stream)
        mat_mask = cv2.imread(mat_path, cv2.IMREAD_GRAYSCALE)
        scale_factor = 1  # 缩放因子
        tx, ty = 0, 0  # 水平平移，垂直平移 
        current_overlap_loss = self.overlap(stream_mask, mat_mask, scale_factor,tx,ty)
        learning_rate = 0.1  # "学习率"
        unit_offset = 5  # 单位平移量
        sum_pix=self.width*self.height
        max_iter = 500  # 最大迭代次数
        loss_record = [current_overlap_loss]
        x_record = [tx]
        y_record = [ty]
        scale_record = [scale_factor]
        clear_folder('optVis')
        for i in range(max_iter):
            scale_loss = self.overlap(stream_mask, mat_mask, scale_factor*(1+learning_rate),tx,ty) - current_overlap_loss
            tx_loss = self.overlap(stream_mask, mat_mask, scale_factor,tx+unit_offset,ty) - current_overlap_loss
            ty_loss = self.overlap(stream_mask, mat_mask, scale_factor,tx,ty+unit_offset) - current_overlap_loss
            scale_factor = scale_factor*(1 - learning_rate*scale_loss/sum_pix)
            tx = tx - learning_rate*tx_loss/sum_pix
            ty = ty - learning_rate*ty_loss/sum_pix
            current_overlap_loss = self.overlap(stream_mask, mat_mask, scale_factor,tx,ty,True)
            loss_record.append(current_overlap_loss)
            x_record.append(tx)
            y_record.append(ty)
            scale_record.append(scale_factor)
        end_time = time.time()
        print(f"优化用时：{end_time-start_time}s")
        print(f"scale_factor: {scale_factor}, tx: {tx}, ty: {ty}")
        self.images_to_video('optVis', 60)
        plt.plot(loss_record)
        plt.title("loss")
        plt.show()
        plt.plot(x_record)
        plt.plot(y_record)
        plt.title("tx,ty")
        plt.show()
        plt.plot(scale_record)
        plt.title("scale")
        plt.show()

    def optim_mask_with_contour(self,stream,mat_contour):
        start_time = time.time()
        stream_mask,stream_contour = self.plot_polygon(stream)
        Polygon_canvas=Polygon([(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)])
        Polygon_mat =  Polygon(mat_contour)
        Polygon_stream = Polygon(stream_contour[0])
        centroid_mat = Polygon_mat.centroid
        centroid_stream = Polygon_stream.centroid
        # tx=centroid_stream.x - centroid_mat.x
        # ty=centroid_stream.y - centroid_mat.y
        scale_factor = 0.3
        min_x, min_y, max_x, max_y = Polygon_stream.bounds
        tx_list=np.linspace(min_x,max_x,3).tolist()
        ty_list=np.linspace(min_y,max_y,3).tolist()
        tx_list=[int(i)-centroid_mat.x for i in tx_list]
        ty_list=[int(i)-centroid_mat.y for i in ty_list]
        clear_folder('optVis')
        final_loss_record = []
        final_transform_record = []
        for tx in tx_list:
            for ty in ty_list:
                scale_factor_record = [scale_factor]#list, for recording
                tx_record = [tx]#list, for recording
                ty_record = [ty]#list, for recording
                translated_scaled_polygon_mat = self.translate_scale_polygon(Polygon_mat,tx,ty,scale_factor)
                loss_record = []
                unit_offset = 25
                unit_scale_rate = 0.1
                scale_learning_rate = 0.05
                translate_learning_rate = 1500
                sum_pix=self.width*self.height
                for i in range(100):
                    current_overlap_loss = self.overlap_polygon(Polygon_stream,translated_scaled_polygon_mat)
                    if i>30 and loss_record[-1]>0.99*0.5*(loss_record[-2]+loss_record[-3]):
                        print("early stop,i: ",i)
                        break
                    x_loss = self.overlap_polygon(Polygon_stream,self.translate_scale_polygon(translated_scaled_polygon_mat,unit_offset,0,1)) - current_overlap_loss
                    y_loss = self.overlap_polygon(Polygon_stream,self.translate_scale_polygon(translated_scaled_polygon_mat,0,unit_offset,1)) - current_overlap_loss
                    scale_loss = self.overlap_polygon(Polygon_stream,self.translate_scale_polygon(translated_scaled_polygon_mat,0,0,1+unit_scale_rate)) - current_overlap_loss
                    tx = - translate_learning_rate*x_loss/sum_pix
                    ty = - translate_learning_rate*y_loss/sum_pix
                    tx_record.append (tx)
                    ty_record.append (ty)
                    scale_factor=(1-scale_learning_rate*scale_loss/sum_pix)
                    scale_factor_record.append(scale_factor)
                    loss_record.append(current_overlap_loss)
                    translated_scaled_polygon_mat = self.translate_scale_polygon(translated_scaled_polygon_mat,tx,ty,scale_factor)
                # tx,ty可累加，表示pollygon.centroid的移动，我们取polygon中最小的x和y，便于后面绘制图像，scale操作每次都以当时的centroid为中心，故可以直接累乘
                import math
                min_x , min_y,_,_=translated_scaled_polygon_mat.bounds
                transform=[int(min_x),int(min_y),math.prod(scale_factor_record)]
                end_time = time.time()
                final_loss_record.append(loss_record[-1])
                final_transform_record.append(transform)
                print(f"优化用时：{end_time-start_time}s")
                print(f"transform: {transform}")
                print(f"loss: {loss_record[-1]}")
                
        best_trans_index= final_loss_record.index(min(final_loss_record))
        best_trans=final_transform_record[best_trans_index]       
        # self.images_to_video('optVis', 60)
        # plt.plot(loss_record)
        # plt.title("loss")
        # plt.show()
        # plt.plot(tx)
        # plt.plot(ty)
        # plt.title("tx,ty")
        # plt.show()
        # plt.plot(scale_factor)
        # plt.title("scale")
        # plt.show()
        return best_trans
    
    def get_tree_name(self):
        name1="shu3.png"
        name2="shu2.png"
        name_list = [name1,name2]
        return name_list
    
    def paste_tree(self,tree_name,img):
        if len(tree_name)!=len(self.ves):
            raise ValueError("the number of trees is not equal to the number of ves")
        for i in range(len(self.ves)):
            ves = self.ves[i]
            tree_hight = math.dist(ves[0],ves[1])
            if ves[0][0]==ves[1][0]:
                theta = math.pi/2
            else:
                k,b = calculate_linear_function(ves[0],ves[1])
                theta = math.atan(k)
            if theta<0:
                theta+=math.pi
            theta_degree = theta/math.pi*180
            tmp_img = Image.open('data/images/'+tree_name[i]).convert("RGBA")
            scale_factor=tree_hight/tmp_img.height
            tmp_img = tmp_img.resize((int(tmp_img.width*scale_factor),int(tmp_img.height*scale_factor)))
            tmp_img.rotate(theta_degree-90,expand=True)#旋转中心取默认的图像中心，最后令图像中心与ves的中点重合
            img.paste(tmp_img,(int((ves[0][0]+ves[1][0])/2-tmp_img.width/2),int((ves[0][1]+ves[1][1])/2-tmp_img.height/2)),mask=tmp_img)
    
    def match_contour(self,img_contour,contours):
        """
        match the image with the contours
        return the best match
        """
        best_record = 0
        best_record_key = None
        best_record_contour = None
        for key, contour in contours.items():
            match_score = cv2.matchShapes(img_contour, contour, cv2.CONTOURS_MATCH_I3, 0)#the pram can be changed to other match methods
            print(f"match score: {match_score}")
            if match_score > best_record:
                best_record = match_score
                best_record_key = key
                best_record_contour = contour
        print(f"best match: {best_record_key}")
        return best_record_key, best_record_contour
    
    def calc_contours(self,path):
        """
        calculate the contours of the image in the folder
        """
        contour_dict = {}
        for filename in os.listdir(path):
            if filename.endswith('.png'):
                img_path = os.path.join(path,filename)
                tmp_contour = get_contour(img_path)#get the approximate contour of the image
                contour_dict[filename] = tmp_contour
        return contour_dict

    def draw_overlap(self,image1, image2):
        # 红色图片：灰度值映射到红色通道，透明度 50%
        red_image = np.zeros((image1.shape[0], image1.shape[1], 3), dtype=np.uint8)
        red_image[:, :, 2] = image1  # 红色通道
        red_image = cv2.addWeighted(red_image, 0.5, np.zeros_like(red_image), 0.5, 0)  # 透明度 50%

        # 蓝色图片：灰度值映射到蓝色通道，透明度 50%
        blue_image = np.zeros((image2.shape[0], image2.shape[1], 3), dtype=np.uint8)
        blue_image[:, :, 0] = image2  # 蓝色通道
        blue_image = cv2.addWeighted(blue_image, 0.5, np.zeros_like(blue_image), 0.5, 0)  # 透明度 50%

        # 重叠图片
        overlay_image = cv2.addWeighted(red_image, 1, blue_image, 1, 0)

        # 保存结果
        output_folder = "optVis"
        os.makedirs(output_folder, exist_ok=True)  # 创建文件夹（如果不存在）
        # 根据现存文件数命名
        existing_files = os.listdir(output_folder)
        new_file_name = f"overlay_{len(existing_files):05}.png"
        output_path = os.path.join(output_folder, new_file_name)

        # 保存图片
        cv2.imwrite(output_path, overlay_image)

    def images_to_video(self, folder_name, fps=60):
        """
        将指定文件夹中的所有图片连缀生成一段视频。

        参数:
            folder_name (str): 图片文件夹路径。
            fps (int): 视频帧率（每秒帧数）。

        返回:
            str: 生成的视频文件路径。
        """
        # 获取所有图片文件
        images = [img for img in os.listdir(folder_name) if img.endswith(".png")]
        images.sort()  # 按文件名排序

        # 检查是否有图片
        if not images:
            raise ValueError("未找到图片文件，请检查文件夹路径")

        # 读取第一张图片以获取分辨率
        first_image_path = os.path.join(folder_name, images[0])
        frame = cv2.imread(first_image_path)
        height, width, layers = frame.shape

        # 视频输出路径
        output_video_path = os.path.join(folder_name, "animation.mp4")

        # 设置视频参数
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码格式
        video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # 将每张图片写入视频
        for image in images:
            image_path = os.path.join(folder_name, image)
            frame = cv2.imread(image_path)
            video.write(frame)  # 写入帧

        # 释放视频资源
        video.release()


def clear_folder(folder_path):
    """
    删除文件夹内的所有元素（包括文件和子文件夹）。

    参数:
        folder_path (str): 文件夹路径。
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        raise ValueError(f"文件夹不存在: {folder_path}")

    # 遍历文件夹内的所有元素
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)

        # 如果是文件，直接删除
        if os.path.isfile(item_path):
            os.remove(item_path)

    print(f"文件夹已清空: {folder_path}")














if __name__ == '__main__': 
    if not os.path.exists("./runs"):
        os.mkdir('./runs/')
    
    task_bef = len(os.listdir("./runs/"))
    base_dir = f'./runs/exp{task_bef}/'
    os.mkdir(base_dir)

    args = parse_arguments()
    canvas = Canvas(args)
    canvas.add_background_stream()

    canvas.draw()
    canvas.save_(f'{base_dir}result1.png')


    print(f'平衡度为：{canvas.bias_h}, {canvas.dense_y}')
    print(f"dense= {canvas.dense}")
    canvas.add_background_stream()
    canvas.draw()
    canvas.save_(f'{base_dir}result2.png')
    print(f'平衡度为：{canvas.bias_h}, {canvas.dense_y}')
    print(f"dense= {canvas.dense}")
    canvas.add_background_stream()
    print(f'平衡度为：{canvas.bias_h}, {canvas.dense_y}')
    print(f"dense= {canvas.dense}")
    canvas.draw()

    canvas.save_(f'{base_dir}result_bef.png')

    canvas.clean()
    canvas.segment_stream()
    canvas.add_tree()

    canvas.add_verts(canvas.bgs[-1])
    for _ in range(5):
        canvas.add_foreground_stream()
    canvas.add_peaks()
    canvas.update_bias()


    canvas.draw()

    canvas.save_(f'{base_dir}{args.filename}')

    # ply = [[1260,407],
    #     [ 668  ,  0],
    #     [ 479 ,  78],
    #     [   0 , 761],
    #     [1094 , 761]
    # ]
    # stream = [(327, 0), (591, 121), (875, 374), (982, 450)]
    # canvas.optim_stream_and_ply(stream,ply)
    contour_dict = canvas.calc_contours('data/mask/mountains/')
    bg_transform=[]
    fg_transform=[]
    for bg in canvas.bgs:
        _,bg_contour = canvas.plot_polygon(bg)#bg转化为封闭图形
        best_contour_key,best_contour = canvas.match_contour(bg_contour,contour_dict)#匹配最佳素材，返回素材名称和contour
        best_transform=canvas.optim_mask_with_contour(bg,best_contour)#返回素材优化的位置和放缩
        bg_transform.append({best_contour_key:best_transform})#字典存储素材名称和transform
    for fg in canvas.fgs:
        _,fg_contour = canvas.plot_polygon(fg)
        best_contour_key,best_contour = canvas.match_contour(fg_contour,contour_dict)
        best_transform=canvas.optim_mask_with_contour(fg,best_contour)
        fg_transform.append({best_contour_key:best_transform})
    print(bg_transform)
    print(fg_transform)
    new_img=Image.open('data/images/paper_texture.png').resize((canvas.width,canvas.height))#纹理图像本身大小为1600*450
    if True:
        for i in range(len(canvas.bgs)):
            tmp_img = Image.open('data/images/'+list(bg_transform[i].keys())[0]).convert("RGBA")
            scale_factor=list(bg_transform[i].values())[0][2]
            tmp_img = tmp_img.resize((int(tmp_img.width*scale_factor),int(tmp_img.height*scale_factor)))
            new_img.paste(tmp_img,(list(bg_transform[i].values())[0][0],list(bg_transform[i].values())[0][1]),mask=tmp_img)

        for i in range(len(canvas.fgs)):
            tmp_img = Image.open('data/images/'+list(fg_transform[i].keys())[0]).convert("RGBA")
            scale_factor=list(fg_transform[i].values())[0][2]
            tmp_img = tmp_img.resize((int(tmp_img.width*scale_factor),int(tmp_img.height*scale_factor)))
            new_img.paste(tmp_img,(list(fg_transform[i].values())[0][0],list(fg_transform[i].values())[0][1]),mask=tmp_img)
        tree_name=canvas.get_tree_name()
        canvas.paste_tree(tree_name, new_img)
        new_img.save(f'{base_dir}result_paste.png')
    # canvas.optim_mask(canvas.bgs[-1],'data/mask/mountains/mask_shan1.png')