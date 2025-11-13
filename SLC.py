import cv2 as cv
import numpy as np
img = cv.imread(r"image\University.jpg")
# cv.imshow("Image",img)
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
resize = cv.resize(gray,(500,500))
# cv.imshow("gray",gray)

# Laplacian method 
# only magnituded 
# lap = cv.Laplacian(resize,cv.CV_64F) # Second derivative  may be +ve or -ve
# lap = np.uint8(np.absolute(lap))
# cv.imshow("Laplacian",lap)


# sobel method 
# x axie and y axie we calculated


sobel_x = cv.Sobel(resize,cv.CV_64F,1,0) # x = 1 , y = 0
sobel_y = cv.Sobel(resize,cv.CV_64F,0,1) # x = 0, y = 1

# cv.imshow("Sobel-x",sobel_x)
# cv.imshow("Sobel-y",sobel_y)

# combined_soble = cv.bitwise_or(sobel_x,sobel_y)
# # cv.imshow("Combine",combined_soble)
# combined_soble = np.uint8(np.absolute(combined_soble))

# cv.imshow("Combined_soble",combined_soble)


# canny 
edges = cv.Canny(resize,threshold1=100,threshold2=300)
cv.imshow("Canny",edges)

cv.waitKey(0)

