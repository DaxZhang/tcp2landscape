
import numpy as np
import cv2

def custom_pixel_access(image, x, y, channels):
    """
    自定义的像素访问函数, 用于获取图像中指定坐标的像素值, 
    当坐标超出边界时, 从图像另一侧获取像素值。

    参数: 
    image: 输入的图像(以numpy数组形式表示)
    x: 要访问的像素的横坐标
    y: 要访问的像素的纵坐标
    channels: 图像的通道数(例如, 对于RGB图像为3, 灰度图像为1)

    返回值: 
    返回对应坐标处的像素值(以numpy数组形式表示, 如果是多通道则是相应维度的数组)
    """
    height, width = image.shape[:2]
    # 处理横坐标超出边界的情况, 通过取模操作实现环绕
    x = x % width
    # 处理纵坐标超出边界的情况, 通过取模操作实现环绕
    y = y % height
    if channels == 1:
        return image[y, x]
    else:
        return image[y, x, :]



def calculate_hu_moments(points):
    points = np.array(points)
    # 连接起点和终点形成封闭多边形
    points = np.vstack((points, points[0]))
    contour = np.array([points])
    moments = cv2.moments(contour)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments

def similarity_between_poly_and_line(polygon_points,target_points):
    p_humoment = calculate_hu_moments(polygon_points)
    t_humoment = calculate_hu_moments(target_points)
    return np.sum(np.abs(p_humoment-t_humoment))


def match_between_poly_and_line(polygon_points, target_points):
    rounded_polygon_points = np.vstack((polygon_points,polygon_points))
    n_polygon = len(polygon_points)
    best_metric = float('inf')
    best_be = [0,0]
    for i in range(n_polygon):
        for j in range(i+1,i+n_polygon):
            cost = similarity_between_poly_and_line(rounded_polygon_points[i:j],target_points)
            if cost<best_metric:
                best_metric = cost
                best_be[0] = i
                best_be[1] = j    

    start_p = np.array(rounded_polygon_points[i])
    end_p = np.array(rounded_polygon_points[j-1])
    
    tar_start_p = np.array(target_points[0])
    tar_end_p = np.array(target_points[-1])

    T = tar_start_p -start_p
    transformed_end_p = T + end_p
    S = np.linalg.norm(tar_end_p-tar_start_p)/np.linalg.norm(transformed_end_p-tar_start_p)

    scaled_transformed_end_p = tar_start_p + S * (transformed_end_p - tar_start_p)
    origin_v = scaled_transformed_end_p - tar_start_p
    target_v = tar_end_p - tar_start_p

    theta = np.arccos(np.dot(origin_v,target_v)/(np.linalg.norm(target_v),np.linalg.norm(target_v)))
    
    if np.cross(origin_v,target_v) < 0:
        theta = -theta

    R = [
        [np.cos(theta),-np.sin(theta)],
        [np.sin(theta),np.cos(theta)]
    ]

    pin = tar_start_p
    return T, S, R, pin,theta # translate, scale, rotate

    

def paste_image(foreground,background,mask,T,S,R,pin,theta):
    x_offset = T[0]
    y_offset = T[1]

    M = np.float32([[1, 0, x_offset], [0, 1, y_offset]])
    translated_foreground = cv2.warpAffine(foreground, M, (foreground.shape[1], foreground.shape[0]))
    translated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

    rows, cols = translated_foreground.shape[:2]

    # 定义缩放中心和缩放大小
    center_x, center_y = pin[0],pin[1]
    scale = S

    # 计算平移矩阵, 将缩放中心平移到图像左上角
    M_translate_to_origin = np.float32([[1, 0, -center_x], [0, 1, -center_y]])
    # 计算缩放矩阵
    M_scale = np.float32([[scale, 0, 0], [0, scale, 0]])
    # 计算平移矩阵, 将缩放后的图像平移回原来的中心位置
    M_translate_back = np.float32([[1, 0, center_x], [0, 1, center_y]])

    # 组合变换矩阵
    M = M_translate_back.dot(M_scale.dot(M_translate_to_origin))
    # 进行缩放操作
    scaled_image = cv2.warpAffine(translated_foreground, M, (cols, rows))
    scaled_mask = cv2.warpAffine(translated_mask,M,(cols,rows))

    cv2.imshow('test',scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rows, cols = scaled_image.shape[:2]
    M = cv2.getRotationMatrix2D((pin[1], pin[0]), theta, 1)
    rotated_foreground = cv2.warpAffine(scaled_image, M, (cols, rows))
    rotated_mask = cv2.warpAffine(scaled_mask,M,(cols,rows))

    cv2.imshow('test',rotated_foreground)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    height, width = rotated_mask.shape[:2]

    for y in height:
        for x in width:
            if rotated_mask[y,x] != 0:
                background[y,x] = rotated_foreground[y,x]
    
    cv2.imshow('test',background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return background

    

    



