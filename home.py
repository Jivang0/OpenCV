import cv2 as cv
import numpy as np

img = cv.imread(r"image\Downfall.jpg")
print("Image Dimension",img.shape)
h,w,c = img.shape

# resized =cv.resize(img,(200,100))
print("Height:",h)
print("width:",w)
print("channel:",c)
# cv.imshow("Image",resized)
# cv.waitKey(0)