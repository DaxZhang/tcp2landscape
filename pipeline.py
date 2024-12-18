import cv2
import contour
import match
from stream import *



if __name__ == '__main__':
    num_background = 2
    num_foreground = 1
    num_pulse = 40
    draft, pulse_pts, background_pts, foreground_pts = generate_line_segments(num_background,num_pulse,num_foreground)


    seg_background_pts = []
    for i in range(len(background_pts)-1):
        seged = segment_stream(background_pts[i+1])
        for seg in seged:
            seg_background_pts.append(seg)

    seg_foreground_pts = []
    for i in range(len(foreground_pts)):
        seged = segment_stream(foreground_pts[i])
        for seg in seged:
            seg_foreground_pts.append(seg)

    # print('seg',seg_background_pts)
    # print('seg',seg_foreground_pts)
    pulse_pts = generate_peak(background_pts[0],num_pulse)

    trees = []
    trees.append(generate_vert_line(seg_foreground_pts[0]))
    trees.append(generate_vert_line(seg_background_pts[0],up = False))


    image = draw(1600,450,seg_background_pts,seg_foreground_pts,pulse_pts,trees)

    image.save("image.png")
    


    
