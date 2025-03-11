import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_std_with_alpha(channel_flatten,mask):
    valid_pixels = channel_flatten[mask]
    sum_ = np.sum(valid_pixels)
    count = valid_pixels.size  # 等价于 np.count_nonzero(mask)
    mean = sum_ / count
    var = np.sum((valid_pixels - mean) ** 2)
    std = np.sqrt(var)
    print("mean:",mean,"std:",std)
    return mean,std

def match_color(img,img_ref):
    # 分离通道
    b,g,r,a=cv2.split(img)
    channel=[b,g,r,a]
    mask = a != 0
    b,g,r,a=cv2.split(img_ref)
    channel_ref=[b,g,r,a]
    mask_ref = a != 0
    for i in range(0,3):
        mean,std=mean_std_with_alpha(channel[i],mask)
        mean_ref,std_ref=mean_std_with_alpha(channel_ref[i],mask_ref)
        channel[i]=(channel[i]-mean)/std*std_ref+mean_ref
        channel[i]=channel[i].astype(np.uint8)
    merge=cv2.merge([channel[0],channel[1],channel[2],channel[3]])
    cv2.imwrite("color_match.png", merge)
    return merge

def match_color2(img,img_ref):
    b,g,r,a=cv2.split(img)
    channel=[b,g,r,a]
    mask = a != 0
    b,g,r,a=cv2.split(img_ref)
    channel_ref=[b,g,r,a]
    mask_ref = a != 0

    for i in range(0,3):
        hist = cv2.calcHist([channel[i]],[0],channel[3],[256],ranges=[0, 256])
        hist_ref = cv2.calcHist([channel_ref[i]],[0],channel_ref[3],[256],ranges=[0, 256])
        source_cdf = np.cumsum(hist)/hist.sum()
        ref_cdf = np.cumsum(hist_ref)/hist_ref.sum()
        mapping = np.interp(source_cdf,ref_cdf,np.arange(256))
        channel[i] = np.round(mapping[channel[i]]).astype(np.uint8)
    merge=cv2.merge([channel[0],channel[1],channel[2],channel[3]])
    cv2.imwrite("color_match2.png",merge)
    return merge
    #     print(type(hist))
    #     print(hist.shape)
    #     print(hist)

    # print(hist)
# 读取彩色图像 (BGR格式)
img = cv2.imread('./data/images/shan10.png',cv2.IMREAD_UNCHANGED)
img_ref=cv2.imread('./data/images/shan3.png',cv2.IMREAD_UNCHANGED)
# 检查图像是否加载成功
if img is None or img_ref is None:
    print("Error: 无法读取图像，请检查文件路径")
    exit()
match_color(img,img_ref)

# 显示图像
# plt.imshow(cv2.cvtColor(merge, cv2.COLOR_BGRA2RGBA))

# # 计算各通道直方图
# hist_b = cv2.calcHist([b], [0],None,[256], [10, 256])
# hist_g = cv2.calcHist([g], [0],None,[256], [10, 256])
# hist_r = cv2.calcHist([r], [0],None,[256], [10, 256])
# hist_a = cv2.calcHist([a], [0],None,[256], [0, 256])
# for i in range (0,256):
#     if i==0:
#         hist_b[i].append(hist_b[i][0])
#     else:
#         hist_b[i].append(hist_b[i][0]+hist_b[i-1][1])


# print(type(hist_a))
# print(hist_a.shape)
# print(hist_a.dtype)
# print('type',type(hist_a[0][0]))
# # # 创建画布
# plt.figure(figsize=(12, 6))

# # 绘制直方图
# plt.plot(hist_b, color='blue', alpha=0.7, label='Blue')
# plt.plot(hist_g, color='green', alpha=0.7, label='Green')
# plt.plot(hist_r, color='red', alpha=0.7, label='Red')
# # 添加标签和标题
# plt.title('Color Histogram')
# plt.xlabel('Pixel Value')
# plt.ylabel('Frequency')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.5)

# # 显示结果
# plt.show()