import cv2 as cv

face_casade = cv.CascadeClassifier(cv.data.haarcascades+"haarcascade_frontalface_default.xml")

cap = cv.VideoCapture(0) # or 1
while True:
    tr,frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)

    faces = face_casade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors = 5)

    for (x,y,w,h) in faces:
        cv.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
        cv.imshow("Face Detection",frame)
        

    if cv.waitKey(1)& 0xFF == ord('q'):
      break



cap.release()
cv.destroyAllWindows()









# while True:
#     set_f,frame = cap.read()
#     cv.imshow("Face",frame)

#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break