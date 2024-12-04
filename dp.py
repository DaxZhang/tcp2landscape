
import numpy as np
import cv2



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


    

    

