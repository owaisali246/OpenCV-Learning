import math
import time
import cv2
import numpy as np

# ----------------- Image Basics -----------------

# black_bg = np.ones((500, 500), np.uint8) * 0  # for black background
# white_bg = np.ones((500, 500), np.uint8) * 255  # for white background
# cv2.imshow('White Screen',white_bg)
# cv2.imshow('Black Screen',black_bg)

# img1 = cv2.imread("assets/img-2.jpg", 0)  # 0 for grayscale
# img2 = cv2.imread("assets/img-2.jpg", 1)  # 1 for color
# img3 = cv2.imread("assets/img-2.jpg", -1)  # -1 for unchanged

# cv2.imshow("Original Image", img1)

# key = cv2.waitKey(0)
# if key == 27:
#     cv2.destroyAllWindows()
# elif key == ord("s"):
#     cv2.imwrite("assets/img-2-copy.jpg", img1) # to save the image
#     cv2.destroyAllWindows()


# --------------- Video Capture ---------------

# cap = cv2.VideoCapture(0)  # 0 for default camera
# # print (cap.get(cv2.CAP_PROP_FRAME_WIDTH)) # to get the width of the frame
# # print (cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # to get the height of the frame
# print (cap.get(cv2.CAP_PROP_FPS)) # to get the fps of the video

# fourcc = cv2.VideoWriter_fourcc(*"XVID") # to get the codec
# out = cv2.VideoWriter("assets/output.avi", fourcc, 30.0, (640, 480)) # to save the video

# while (cap.isOpened()):
#     ret, frame = cap.read() # ret is boolean, frame is the image
    
#     if ret == True:
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # for grayscale video
#         out.write(frame) # to save the video
#         cv2.imshow("Live Video", gray)
#         if cv2.waitKey(1) == ord("q"):
#             break
#     else:
#         break
    

# cap.release()
# out.release() # to save the video
# cv2.destroyAllWindows()


# --------------- Drawing Shapes ---------------
def draw_shapes():
    img = np.ones((500, 500, 3), dtype=np.uint8) * 0
    cv2.line(img, (250,150), (250,350), (0, 255, 255), math.floor(10)) # to draw a line
    cv2.arrowedLine(img, (100,200), (350,200), (205, 230, 0), 10) # to draw an arrowed line
    cv2.rectangle(img, (100, 100), (400, 400), (0, 255, 0), 10) # to draw a rectangle
    cv2.circle(img, (250, 250), 100, (255, 0, 0), -1) # to draw a circle
    cv2.putText(img, "OpenCV", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5) # to put text
    cv2.imshow("Image", img)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()

def draw_logo():
    main_radius, mini_radius = 70, 30
    blue, green, red, black = (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)
    pt1, pt2, pt3 = (250, 150), (170, 290), (330, 290)
    img = np.ones((500, 500, 3), np.uint8) * 0
    cv2.circle(img, pt1, main_radius, red, -1) # to draw a circle
    cv2.circle(img, pt1, mini_radius, black, -1) # to draw a circle
    cv2.circle(img, pt2, main_radius, green, -1) # to draw a circle
    cv2.circle(img, pt2, mini_radius, black, -1) # to draw a circle
    cv2.drawContours(img, [np.array([pt1, pt2, pt3])], 0, black, -1)
    cv2.circle(img, pt3, main_radius, blue, -1) # to draw a circle
    cv2.circle(img, pt3, mini_radius, black, -1) # to draw a circle
    cv2.drawContours(img, [np.array([(290,220), (370,220), pt3])], 0, black, -1)
    # cv2.drawContours(img, [np.array([(), (), pt3])], 0, black, -1)
    cv2.putText(img, "OpenCV", (130, 430), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5) # to put text
    cv2.imshow("Line", img)
    if cv2.waitKey(0) == ord('q'):
        cv2.destroyAllWindows()


def draw_animation():
    for i in range (0, 100):
        img = np.ones((500, 500, 3), np.uint8) * 255
        cv2.line(img, (250-i,150+i), (250+i,350-i), (0, 255-i, 255-i), math.floor(10)) # to draw a line
        cv2.arrowedLine(img, (100,200), (350,200), (205, 30, 0+i), 10) # to draw an arrowed line
        cv2.imshow("Line", img)
        if cv2.waitKey(10) == ord('q'):
            break
    cv2.destroyAllWindows()

