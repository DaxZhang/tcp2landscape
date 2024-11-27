#xiao wu
#coding:utf-8
 
 
import os
import cv2
import glob
from PIL import Image
import numpy as np
import matplotlib.image as mpimg


data_dir = './data/image/'
output_dir = './data/images/'
mask_dir = './data/mask'



 
#位深度判断&转换#去除蒙版
for filename in os.listdir(data_dir):  
    #cv模块读取图像信息并打印                        
    img=cv2.imread(os.path.join(data_dir,filename),cv2.IMREAD_UNCHANGED)
    print('cv2 img_shape',img.shape)
 
    #matplotlib模块读取图像信息并打印
    img=mpimg.imread(os.path.join(data_dir,filename))
    print('matplotlib img_shape',img.shape)
 
    #去除背景信息
    img = Image.open(os.path.join(data_dir,filename))
    img = img.convert('RGBA')
    pixdata = img.load()
    add_mask = Image.new("RGB",img.size,(255,255,255))
    mask = Image.new("L",img.size,0)
    add_maskdata=add_mask.load()
 
    #逐个像素读取并处理
    for y in range(img.size[1]):
        print()
        for x in range(img.size[0]):
            # print(pixdata[x, y][3], end=' ')
            if pixdata[x, y][0] > 0 and pixdata[x, y][1] > 0 and pixdata[x, y][2] > 0 and pixdata[x, y][3] == 255:
                add_mask.putpixel((x,y),pixdata[x, y][0:3])
                mask.putpixel((x,y),pixdata[x, y][3])

    add_mask.save(os.path.join(output_dir,filename))
    mask.save(os.path.join(mask_dir,'mask_'+filename))