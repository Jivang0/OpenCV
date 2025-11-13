import cv2 as cv
# Reading images
img =cv.imread(r"image\$50,000-Picsart-AiImageEnhancer.png") 
# cv.imshow("Image",img)

#cv.waitkey(3000) # 3000 bnako 3 sec 
# cv.waitKey(3000) # any key press nagarda image display huncha but key press jancha

resized = cv.resize(img,(100,100)) # img lahi resize 200*200 pixel
# cv.imshow("Image",resized)#Image title ma aahucha  
# cv.waitKey(0)

# flipped = cv.flip(resized,1) # 0 vertically, 1 horizontal, -1 for both vert & horizontal

# gray = cv.cvtColor(resized,cv.COLOR_BGR2GRAY) # gray sclae 2 dimension black and white(1,0), 
# cv.imshow("Flip",gray)
# print(gray.shape)


# cv.rectangle(img,(400,250),(650,650),(0,255,0),2)
# (100,100) banko top point and (500,500) bottom point
# (0,255,0) banko BGR and 5 is the thickness


# edge detection 
# canyy = ml model = pre_trained

# edges = cv.Canny(img,threshold1 = 100, threshold2 = 200)
# cv.imshow("Edges",edges)


# Haar Cascade 
import cv2 as cv
# Reading images
img =cv.imread(r"image\$50,000-Picsart-AiImageEnhancer.png") 
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors = 5,minSize=(60,60))
print("Number of faces detected:",len(faces))
# ((x,y),w,h),,
for ( x,y,w,h) in faces:
    cv.rectangle(img,(x,y),(x + w ,y+h),(0,255,0),2)
cv.imshow("Faces",img)


cv.waitKey(0)




