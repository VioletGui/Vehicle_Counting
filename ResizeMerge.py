import cv2

img_bg = cv2.imread('background.jpg')
img_fg = cv2.imread('foreground.jpg')

height_b, width_b, channels_b = img_bg.shape
height_f, width_f, channels_f = img_fg.shape

# Set the desired center point for the foreground picture
xc = 130
yc = 480

# Set the desired height and width for the foreground picture
target_height = 300
target_width = 220

resized_img_fg = cv2.resize(img_fg, (target_width, target_height)) # width, height

#height_rf, width_rf, channels_rf = resized_img_fg.shape

img_bg[yc - int(target_height / 2) : yc + int(target_height / 2), xc - int(target_width / 2) : xc + int(target_width / 2)] = resized_img_fg
# height, width

cv2.imshow('mixed_pic', img_bg)
cv2.waitKey(0)
cv2.destroyAllWindows()


#print (height_f, width_f, channels_f)
#print (height_b, width_b, channels_b)

#print (target_height)
#print (target_width)

#print (xc - int(target_width / 2), xc + int(target_width / 2))

#print (yc - int(target_height / 2), yc + int(target_height / 2))

#print (height_rf, width_rf, channels_rf)

