import cv2

img = cv2.imread('img-test.jpg')
x0 = 200
y0 = 300
cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + 2, y0 + 2), (255, 0, 0), -1)
cv2.imshow('CenterDrawRect', img)
cv2.imwrite('CenterDrawRect.png', img)

cv2.waitKey(0)

cv2.destroyAllWindows()
