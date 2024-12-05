import cv2
import contour
import match
from stream import *



if __name__ == '__main__':
    num_background = 2
    num_foreground = 1
    num_pulse = 20
    draft, pulse_pts, background_pts, foreground_pts = generate_line_segments(num_background,num_pulse,num_foreground)

    for i in range(len(background_pts)):
        background_pts[i] = segment_stream(background_pts[i])
        print(background_pts[i])

    for i in range(len(foreground_pts)):
        foreground_pts[i] = segment_stream(foreground_pts[i])

    image = draw(1600,450,background_pts,foreground_pts,pulse_pts)

    image.save("image.png")
    


    
