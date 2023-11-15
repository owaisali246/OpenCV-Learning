import cv2
import numpy as np

white_bg = np.ones((500, 500), np.uint8)*0
cv2.imshow('White Screen',white_bg)

cv2.waitKey(0)
cv2.destroyAllWindows()

