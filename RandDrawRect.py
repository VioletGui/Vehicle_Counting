import cv2
import numpy as np


img = cv2.imread('img-test.jpg')
height, width, channels = img.shape
x1 = np.random.randint(4, width-4)
y1 = np.random.randint(4, height-4)
cv2.rectangle(img, (x1, y1), (x1 + 4, y1 + 4), (255, 0, 0), -1)
cv2.imshow('RandDrawRect', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
